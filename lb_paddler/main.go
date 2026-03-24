package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// Backend represents a PaddleOCR server instance
type Backend struct {
	URL                 string
	Healthy             int32         // atomic: 1=healthy, 0=unhealthy
	ActiveConns         int64         // atomic
	ConsecutiveFailures int           // protected by mu
	CircuitOpenUntil    time.Time     // protected by mu
	mu                  sync.Mutex    // protects ConsecutiveFailures and CircuitOpenUntil
}

func (b *Backend) IsHealthy() bool {
	return atomic.LoadInt32(&b.Healthy) == 1
}

func (b *Backend) SetHealthy(healthy bool) {
	if healthy {
		atomic.StoreInt32(&b.Healthy, 1)
	} else {
		atomic.StoreInt32(&b.Healthy, 0)
	}
}

func (b *Backend) GetActiveConns() int64 {
	return atomic.LoadInt64(&b.ActiveConns)
}

// LoadBalancer manages multiple backends
type LoadBalancer struct {
	backends           []*Backend
	maxConnsPerBackend int
	client             *http.Client
	requestQueue       chan *queuedRequest
	queueSize          int
	queueWorkers       int
	stats              *Stats
}

// Stats tracks request statistics
type Stats struct {
	TotalRequests   int64
	SuccessRequests int64
	FailedRequests  int64
	QueuedRequests  int64
}

type queuedRequest struct {
	w    http.ResponseWriter
	r    *http.Request
	done chan struct{}
	body []byte
}

// Config holds the configuration
type Config struct {
	ListenPort          int      `json:"listen_port"`
	BackendURLs         []string `json:"backend_urls"`
	MaxConnsPerBackend  int      `json:"max_conns_per_backend"`
	QueueSize           int      `json:"queue_size"`
	QueueWorkers        int      `json:"queue_workers"`
	HealthCheckInterval int      `json:"health_check_interval_seconds"`
	RequestTimeout      int      `json:"request_timeout_seconds"`
}

// Circuit Breaker 설정값
const (
	circuitBreakerThreshold = 3                // 연속 실패 횟수 임계값
	circuitBreakerTimeout   = 30 * time.Second // 차단 시간
	maxRetries              = 2                // 최대 재시도 횟수
)

func NewLoadBalancer(config *Config) *LoadBalancer {
	backends := make([]*Backend, len(config.BackendURLs))
	for i, url := range config.BackendURLs {
		backends[i] = &Backend{
			URL:     url,
			Healthy: 1,
		}
	}

	queueWorkers := config.QueueWorkers
	if queueWorkers <= 0 {
		queueWorkers = 4
	}

	lb := &LoadBalancer{
		backends:           backends,
		maxConnsPerBackend: config.MaxConnsPerBackend,
		client: &http.Client{
			Timeout: time.Duration(config.RequestTimeout) * time.Second,
		},
		requestQueue: make(chan *queuedRequest, config.QueueSize),
		queueSize:    config.QueueSize,
		queueWorkers: queueWorkers,
		stats:        &Stats{},
	}

	return lb
}

// getLeastConnBackend returns the backend with least connections
func (lb *LoadBalancer) getLeastConnBackend() *Backend {
	return lb.getLeastConnBackendExcluding(nil)
}

// getLeastConnBackendExcluding returns the backend with least connections, excluding specified backend
// Circuit Breaker: 차단된 백엔드는 제외
func (lb *LoadBalancer) getLeastConnBackendExcluding(exclude *Backend) *Backend {
	var selected *Backend
	var minConns int64 = int64(lb.maxConnsPerBackend + 1)
	now := time.Now()

	for _, backend := range lb.backends {
		if backend == exclude {
			continue
		}

		healthy := backend.IsHealthy()
		conns := backend.GetActiveConns()

		backend.mu.Lock()
		circuitOpen := backend.CircuitOpenUntil.After(now)
		backend.mu.Unlock()

		// Circuit Breaker: 차단 중이면 건너뜀
		if circuitOpen {
			continue
		}

		if healthy && conns < minConns && conns < int64(lb.maxConnsPerBackend) {
			selected = backend
			minConns = conns
		}
	}
	return selected
}

// proxyRequestWithRetry forwards the request to a backend with retry logic
// Returns true if the response was sent to the client, false otherwise
func (lb *LoadBalancer) proxyRequestWithRetry(w http.ResponseWriter, r *http.Request, body []byte) bool {
	var lastBackend *Backend

	for attempt := 0; attempt <= maxRetries; attempt++ {
		backend := lb.getLeastConnBackendExcluding(lastBackend)
		if backend == nil {
			if attempt == 0 {
				// 첫 시도에서 사용 가능한 백엔드가 없음
				return false
			}
			break
		}

		success := lb.proxyRequest(w, r, body, backend)
		if success {
			return true
		}

		lastBackend = backend
		log.Printf("[WARN] Retry %d: Backend %s failed, trying another...", attempt+1, backend.URL)
	}

	// 모든 재시도 실패
	http.Error(w, "All backends failed", http.StatusServiceUnavailable)
	atomic.AddInt64(&lb.stats.FailedRequests, 1)
	return true // 에러 응답이라도 클라이언트에 전송됨
}

// proxyRequest forwards the request to a backend
// Returns true on success, false on failure (for retry logic)
func (lb *LoadBalancer) proxyRequest(w http.ResponseWriter, r *http.Request, body []byte, backend *Backend) bool {
	atomic.AddInt64(&backend.ActiveConns, 1)
	defer atomic.AddInt64(&backend.ActiveConns, -1)

	// Create new request to backend
	targetURL := backend.URL + r.URL.Path
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}

	proxyReq, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		log.Printf("[ERROR] Failed to create proxy request: %v", err)
		return false
	}

	// Copy headers
	for key, values := range r.Header {
		for _, value := range values {
			proxyReq.Header.Add(key, value)
		}
	}

	// Send request to backend
	startTime := time.Now()
	resp, err := lb.client.Do(proxyReq)
	duration := time.Since(startTime)

	if err != nil {
		log.Printf("[ERROR] Backend %s failed: %v (took %v)", backend.URL, err, duration)
		lb.markBackendFailed(backend)
		return false
	}
	defer resp.Body.Close()

	// 성공 시 Circuit Breaker 카운터 리셋
	lb.markBackendSuccess(backend)

	// Copy response headers
	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	// Copy response body
	w.WriteHeader(resp.StatusCode)
	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("[WARN] Failed to copy response body from %s: %v", backend.URL, err)
	}

	atomic.AddInt64(&lb.stats.SuccessRequests, 1)
	log.Printf("[INFO] %s %s -> %s (%d) took %v", r.Method, r.URL.Path, backend.URL, resp.StatusCode, duration)
	return true
}

// markBackendFailed updates backend state on failure (Circuit Breaker)
func (lb *LoadBalancer) markBackendFailed(backend *Backend) {
	backend.mu.Lock()
	defer backend.mu.Unlock()

	backend.ConsecutiveFailures++

	if backend.ConsecutiveFailures >= circuitBreakerThreshold {
		backend.SetHealthy(false)
		backend.CircuitOpenUntil = time.Now().Add(circuitBreakerTimeout)
		log.Printf("[CIRCUIT BREAKER] Backend %s is now OPEN (failures: %d, blocked for %v)",
			backend.URL, backend.ConsecutiveFailures, circuitBreakerTimeout)
	} else {
		log.Printf("[WARN] Backend %s failure %d/%d",
			backend.URL, backend.ConsecutiveFailures, circuitBreakerThreshold)
	}
}

// markBackendSuccess resets backend failure counter on success
func (lb *LoadBalancer) markBackendSuccess(backend *Backend) {
	backend.mu.Lock()
	defer backend.mu.Unlock()

	if backend.ConsecutiveFailures > 0 {
		backend.ConsecutiveFailures = 0
		backend.CircuitOpenUntil = time.Time{}
		backend.SetHealthy(true)
		log.Printf("[CIRCUIT BREAKER] Backend %s recovered, counter reset", backend.URL)
	}
}

// handleRequest handles incoming requests
func (lb *LoadBalancer) handleRequest(w http.ResponseWriter, r *http.Request) {
	atomic.AddInt64(&lb.stats.TotalRequests, 1)

	// Read body once
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}
	r.Body.Close()

	// Try to proxy with retry
	sent := lb.proxyRequestWithRetry(w, r, body)
	if sent {
		return
	}

	// 사용 가능한 백엔드가 없음 → 큐에 넣기
	atomic.AddInt64(&lb.stats.QueuedRequests, 1)
	log.Printf("[WARN] All backends busy, queuing request...")

	queued := &queuedRequest{
		w:    w,
		r:    r,
		body: body,
		done: make(chan struct{}),
	}

	select {
	case lb.requestQueue <- queued:
		// 큐에 들어감, 완료 대기 (클라이언트 연결 끊김도 감지)
		select {
		case <-queued.done:
			// 처리 완료
		case <-r.Context().Done():
			// 클라이언트가 연결을 끊음
			log.Printf("[WARN] Client disconnected while waiting in queue")
		}
	default:
		// Queue full
		http.Error(w, "Service overloaded, please retry later", http.StatusServiceUnavailable)
		atomic.AddInt64(&lb.stats.FailedRequests, 1)
		log.Printf("[ERROR] Queue full (%d/%d), rejecting request", len(lb.requestQueue), lb.queueSize)
	}
}

// processQueue processes queued requests (run as multiple workers)
func (lb *LoadBalancer) processQueue(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			// 셧다운 시 남은 큐 처리
			lb.drainQueue()
			return
		case queued := <-lb.requestQueue:
			lb.processQueuedRequest(ctx, queued)
		}
	}
}

// processQueuedRequest waits for an available backend and processes the request
func (lb *LoadBalancer) processQueuedRequest(ctx context.Context, queued *queuedRequest) {
	// 클라이언트가 이미 끊겼으면 스킵
	select {
	case <-queued.r.Context().Done():
		close(queued.done)
		return
	default:
	}

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	timeout := time.After(time.Duration(300) * time.Second) // 큐 대기 타임아웃

	for {
		select {
		case <-ctx.Done():
			http.Error(queued.w, "Server shutting down", http.StatusServiceUnavailable)
			close(queued.done)
			return
		case <-queued.r.Context().Done():
			// 클라이언트 연결 끊김
			close(queued.done)
			return
		case <-timeout:
			http.Error(queued.w, "Queue timeout", http.StatusGatewayTimeout)
			atomic.AddInt64(&lb.stats.FailedRequests, 1)
			close(queued.done)
			return
		case <-ticker.C:
			backend := lb.getLeastConnBackend()
			if backend != nil {
				lb.proxyRequestWithRetry(queued.w, queued.r, queued.body)
				close(queued.done)
				return
			}
		}
	}
}

// drainQueue rejects remaining queued requests during shutdown
func (lb *LoadBalancer) drainQueue() {
	for {
		select {
		case queued := <-lb.requestQueue:
			http.Error(queued.w, "Server shutting down", http.StatusServiceUnavailable)
			close(queued.done)
		default:
			return
		}
	}
}

// healthCheck periodically checks backend health
func (lb *LoadBalancer) healthCheck(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, backend := range lb.backends {
				go func(b *Backend) {
					checkCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
					defer cancel()

					req, err := http.NewRequestWithContext(checkCtx, http.MethodGet, b.URL+"/", nil)
					if err != nil {
						return
					}

					resp, err := lb.client.Do(req)
					if err != nil {
						b.mu.Lock()
						if b.IsHealthy() {
							log.Printf("[WARN] Backend %s is unhealthy: %v", b.URL, err)
						}
						b.SetHealthy(false)
						b.mu.Unlock()
						return
					}
					resp.Body.Close()

					b.mu.Lock()
					wasHealthy := b.IsHealthy()
					b.SetHealthy(true)
					// Circuit Breaker 상태도 리셋
					b.ConsecutiveFailures = 0
					b.CircuitOpenUntil = time.Time{}
					if !wasHealthy {
						log.Printf("[INFO] Backend %s is now healthy (circuit breaker reset)", b.URL)
					}
					b.mu.Unlock()
				}(backend)
			}
		}
	}
}

// handleStats returns current statistics
func (lb *LoadBalancer) handleStats(w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	stats := map[string]interface{}{
		"total_requests":   atomic.LoadInt64(&lb.stats.TotalRequests),
		"success_requests": atomic.LoadInt64(&lb.stats.SuccessRequests),
		"failed_requests":  atomic.LoadInt64(&lb.stats.FailedRequests),
		"queued_requests":  atomic.LoadInt64(&lb.stats.QueuedRequests),
		"queue_size":       len(lb.requestQueue),
		"queue_capacity":   lb.queueSize,
	}

	backends := []map[string]interface{}{}
	for _, b := range lb.backends {
		b.mu.Lock()
		circuitOpen := b.CircuitOpenUntil.After(now)
		circuitOpenSecs := 0
		if circuitOpen {
			circuitOpenSecs = int(b.CircuitOpenUntil.Sub(now).Seconds())
		}
		backends = append(backends, map[string]interface{}{
			"url":                  b.URL,
			"healthy":              b.IsHealthy(),
			"active_conns":         b.GetActiveConns(),
			"consecutive_failures": b.ConsecutiveFailures,
			"circuit_open":         circuitOpen,
			"circuit_open_secs":    circuitOpenSecs,
		})
		b.mu.Unlock()
	}
	stats["backends"] = backends

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

func loadConfig(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	config := &Config{}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(config); err != nil {
		return nil, err
	}
	return config, nil
}

func main() {
	configPath := flag.String("config", "config.json", "Path to config file")
	flag.Parse()

	// Load configuration
	config, err := loadConfig(*configPath)
	if err != nil {
		log.Printf("Failed to load config from %s, using defaults: %v", *configPath, err)
		config = &Config{
			ListenPort: 18005,
			BackendURLs: []string{
				"http://127.0.0.1:58581",
				"http://127.0.0.1:58582",
				"http://127.0.0.1:58583",
			},
			MaxConnsPerBackend:  2,
			QueueSize:           100,
			QueueWorkers:        4,
			HealthCheckInterval: 5,
			RequestTimeout:      300,
		}
	}

	lb := NewLoadBalancer(config)

	// Context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start queue workers (여러 워커로 병렬 처리)
	for i := 0; i < lb.queueWorkers; i++ {
		go lb.processQueue(ctx)
	}

	// Start health checker
	go lb.healthCheck(ctx, time.Duration(config.HealthCheckInterval)*time.Second)

	// HTTP handlers
	mux := http.NewServeMux()
	mux.HandleFunc("/lb/stats", lb.handleStats)
	mux.HandleFunc("/", lb.handleRequest)

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", config.ListenPort),
		Handler: mux,
	}

	// Graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("[INFO] Shutting down gracefully...")
		cancel()

		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()
		server.Shutdown(shutdownCtx)
	}()

	log.Printf("[INFO] Load Balancer started on port %d", config.ListenPort)
	log.Printf("[INFO] Backends: %v", config.BackendURLs)
	log.Printf("[INFO] Max connections per backend: %d", config.MaxConnsPerBackend)
	log.Printf("[INFO] Queue size: %d, Workers: %d", config.QueueSize, lb.queueWorkers)

	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}