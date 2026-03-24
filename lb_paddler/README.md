# lb_paddler

PaddleOCR 안정성 확보를 위한 Go 기반 로드밸런서

## 배경

PaddleOCR의 layout parser는 GPU 추론 시 동시 요청을 처리하지 못한다. 하나의 인스턴스에 요청이 동시에 들어오면 메모리 corruption이나 segfault로 프로세스가 죽는다:

```
# 메모리 corruption (malloc 오류)
malloc_consolidate(): unaligned fastbin chunk detected

# segfault로 프로세스 사망
[SignalInfo: *** SIGSEGV (@0x0) received by PID 1536538 (TID 0x7c95a2ffd6c0) from PID 0 ***]
```

이를 우회하기 위해 PaddleOCR 인스턴스를 여러 개 띄우고, 로드밸런서가 **인스턴스당 동시 요청을 1개로 제한**하여 순차 처리를 보장한다.

## 아키텍처

```
                         ┌─► PaddleOCR :58581 (GPU 7)
Client ─► LB :18005 ─────┼─► PaddleOCR :58582 (GPU 7)
          (Go)           └─► PaddleOCR :58583 (GPU 7)
```

## 주요 기능

- **Least Connections 로드밸런싱**: 가장 여유있는 백엔드로 분배
- **동시 요청 제한**: 백엔드당 동시 요청 1개로 제한 (설정 가능)
- **요청 큐잉**: 모든 백엔드가 바쁘면 큐에서 대기, 다중 워커로 병렬 처리
- **자동 재시도**: 백엔드 실패 시 다른 백엔드로 최대 2회 재시도
- **Circuit Breaker**: 연속 3회 실패한 백엔드를 30초간 자동 차단 후 복구
- **헬스체크**: 주기적으로 백엔드 상태 확인 및 Circuit Breaker 자동 리셋
- **Graceful Shutdown**: 종료 시 진행 중 요청 완료 후 대기 큐 정리
- **통계 API**: `/lb/stats` 엔드포인트로 상태 확인

## 설정 (config.json)

```json
{
  "listen_port": 18005,
  "backend_urls": [
    "http://127.0.0.1:58581",
    "http://127.0.0.1:58582",
    "http://127.0.0.1:58583"
  ],
  "max_conns_per_backend": 1,
  "queue_size": 100,
  "queue_workers": 4,
  "health_check_interval_seconds": 5,
  "request_timeout_seconds": 300
}
```

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `listen_port` | 18005 | 로드밸런서 리스닝 포트 |
| `backend_urls` | - | PaddleOCR 백엔드 URL 목록 |
| `max_conns_per_backend` | 1 | 백엔드당 최대 동시 요청 수 (PaddleOCR layout parser 제약으로 1 권장) |
| `queue_size` | 100 | 대기 큐 최대 크기 |
| `queue_workers` | 4 | 큐 처리 워커 수 |
| `health_check_interval_seconds` | 5 | 헬스체크 주기 (초) |
| `request_timeout_seconds` | 300 | 백엔드 요청 타임아웃 (초) |

## 사용법

### 1. 빌드
```bash
./build.sh
```

### 2. PaddleOCR 인스턴스 3개 시작
각각 포트 58581, 58582, 58583에서 실행 필요

### 3. 로드밸런서 시작
```bash
./start.sh
```

### 4. 로드밸런서 중지
```bash
./stop.sh
```

### 5. 통계 확인
```bash
curl -s http://localhost:18005/lb/stats | python3 -m json.tool
```

응답 예시:
```json
{
  "total_requests": 150,
  "success_requests": 148,
  "failed_requests": 2,
  "queued_requests": 5,
  "queue_size": 0,
  "queue_capacity": 100,
  "backends": [
    {
      "url": "http://127.0.0.1:58581",
      "healthy": true,
      "active_conns": 1,
      "consecutive_failures": 0,
      "circuit_open": false,
      "circuit_open_secs": 0
    }
  ]
}
```

## 로그 확인
```bash
tail -f lb.log
```

주요 로그 태그:
- `[INFO]` — 정상 요청 처리, 백엔드 복구
- `[WARN]` — 재시도, 큐잉, 클라이언트 끊김
- `[ERROR]` — 큐 만료, 요청 실패
- `[CIRCUIT BREAKER]` — 백엔드 차단/복구

## TODO

- [ ] Prometheus 메트릭 연동