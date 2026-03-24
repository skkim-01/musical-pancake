# 차원 축소 알고리즘 비교 연구 보고서

## 목차

1. [서론: 차원 축소란 무엇인가](#1-서론-차원-축소란-무엇인가)
2. [왜 차원 축소가 필요한가](#2-왜-차원-축소가-필요한가)
3. [차원 축소의 분류 체계](#3-차원-축소의-분류-체계)
4. [PCA (Principal Component Analysis)](#4-pca-principal-component-analysis)
5. [t-SNE (t-distributed Stochastic Neighbor Embedding)](#5-t-sne-t-distributed-stochastic-neighbor-embedding)
6. [UMAP (Uniform Manifold Approximation and Projection)](#6-umap-uniform-manifold-approximation-and-projection)
7. [TriMap](#7-trimap)
8. [PaCMAP (Pairwise Controlled Manifold Approximation)](#8-pacmap-pairwise-controlled-manifold-approximation)
9. [종합 비교 분석](#9-종합-비교-분석)
10. [실무 적용: 2-Stage Pipeline (PCA + UMAP)](#10-실무-적용-2-stage-pipeline-pca--umap)
11. [결론 및 알고리즘 선택 가이드](#11-결론-및-알고리즘-선택-가이드)
12. [참고문헌](#12-참고문헌)

---

## 1. 서론: 차원 축소란 무엇인가

### 1.1 정의

차원 축소(Dimensionality Reduction)란 고차원 데이터를 원래 데이터의 의미 있는 속성을 최대한 보존하면서 저차원 공간으로 변환하는 기법이다. 예를 들어, 1536차원의 텍스트 임베딩 벡터를 사람이 시각적으로 이해할 수 있는 2차원 또는 3차원 좌표로 변환하는 것이 대표적인 활용 사례이다.

### 1.2 직관적 이해

100개의 특성(feature)을 가진 데이터가 있다고 가정해보자. 이것을 100차원 공간의 한 점으로 생각할 수 있다. 인간은 3차원까지만 직관적으로 이해할 수 있기 때문에, 이 100차원의 점들을 2차원 평면 위에 "그림자"처럼 투영해야 한다.

단, 여기서 핵심은 **그림자를 어떤 각도에서 비추느냐**이다. 어떤 각도에서 비추면 겹쳐서 구분이 안 되던 점들이, 다른 각도에서 비추면 깔끔하게 분리되어 보일 수 있다. 차원 축소 알고리즘이란, 결국 "가장 정보량이 많은 각도"를 찾는 방법이다.

### 1.3 임베딩 시각화에서의 역할

현대 NLP에서 텍스트는 임베딩 모델을 통해 고차원 벡터로 변환된다.

| 임베딩 모델 | 벡터 차원 |
|---|---|
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| BGE-M3 | 1024 |
| Voyage-3 | 1024 |
| Cohere Embed v3 | 1024 |

이 벡터들은 의미적 유사성을 반영하지만, 1536차원 공간에서 점들의 분포를 직접 관찰할 수 없다. 차원 축소를 통해 2D로 투영해야만 "비슷한 문서의 청크들이 가까이 모여 있는가", "특정 파일의 청크들이 파편화되지 않았는가"를 시각적으로 판단할 수 있다.

---

## 2. 왜 차원 축소가 필요한가

### 2.1 차원의 저주 (Curse of Dimensionality)

고차원 공간에서는 인간의 저차원 직관이 통하지 않는 여러 현상이 발생한다.

**거리의 집중 현상 (Distance Concentration)**: 차원이 높아질수록, 임의의 두 점 사이의 거리가 점점 비슷해진다. 1536차원에서 임의의 두 점의 유클리드 거리는 거의 일정한 값에 수렴한다. 이는 "가까운 점"과 "먼 점"의 구분이 모호해진다는 것을 의미한다.

수학적으로, d차원에서 n개의 균일분포 점에 대해:

```
max_distance / min_distance → 1  (d → ∞)
```

이 현상 때문에 고차원에서의 유클리드 거리 기반 분석은 신뢰성이 떨어지며, 코사인 유사도와 같은 대안적 거리 척도를 사용하거나, 차원을 축소한 후 분석해야 한다.

**빈 공간의 폭발**: 2차원 단위 정사각형을 10×10 격자로 나누면 100칸이 필요하다. 3차원이면 1,000칸, 10차원이면 10^10(100억) 칸이 필요하다. 같은 양의 데이터로 고차원 공간을 "채우려면" 기하급수적으로 더 많은 데이터가 필요하다.

### 2.2 시각화

인간의 시각 시스템은 2~3차원에 최적화되어 있다. 고차원 데이터의 구조(군집, 이상값, 연속적 변화 등)를 파악하려면 2D 또는 3D로 축소해야 한다.

### 2.3 연산 효율

많은 머신러닝 알고리즘은 차원에 비례하여(또는 그 이상으로) 연산 비용이 증가한다. PCA로 1536차원을 50차원으로 줄이면, 이후 알고리즘(UMAP 등)의 속도가 수 배에서 수십 배 향상될 수 있다.

### 2.4 노이즈 제거

고차원 임베딩의 모든 차원이 동등하게 중요한 것은 아니다. 일부 차원은 유의미한 의미 정보를 담고 있고, 일부는 노이즈에 가깝다. 차원 축소는 중요한 차원만 남기는 효과가 있어, 오히려 데이터의 본질적 구조가 더 선명해지는 경우가 많다.

---

## 3. 차원 축소의 분류 체계

### 3.1 선형 vs. 비선형

| 구분 | 선형 (Linear) | 비선형 (Non-linear) |
|---|---|---|
| 원리 | 선형 변환 행렬로 투영 | 데이터의 곡면(manifold) 구조를 학습 |
| 대표 알고리즘 | PCA, LDA, Random Projection | t-SNE, UMAP, TriMap, PaCMAP |
| 장점 | 빠름, 해석 용이, 재현성 완벽 | 복잡한 구조 포착 가능 |
| 단점 | 비선형 구조 포착 불가 | 느림, 해석 어려움 |

선형 방법은 데이터가 고차원 공간에서 평면(또는 초평면)에 가깝게 분포한다고 가정하고, 비선형 방법은 데이터가 곡면(manifold) 위에 분포한다고 가정한다.

### 3.2 전역 구조 vs. 지역 구조

이것은 차원 축소 알고리즘을 이해하는 데 가장 중요한 개념 중 하나이다.

**전역 구조 (Global Structure)**: 데이터 전체의 거시적 배치. "군집 A와 군집 B 사이의 상대적 거리"가 보존되는가? 멀리 떨어진 점들의 관계가 유지되는가?

**지역 구조 (Local Structure)**: 각 점의 이웃 관계. "이 점의 가장 가까운 10개 이웃이 축소 후에도 가장 가까운 이웃인가?" 가까운 점들의 관계가 유지되는가?

| 알고리즘 | 전역 구조 보존 | 지역 구조 보존 |
|---|---|---|
| PCA | 강함 | 약함 |
| t-SNE | 약함 | 매우 강함 |
| UMAP | 중~강 | 강함 |
| TriMap | 강함 | 강함 |
| PaCMAP | 강함 | 강함 |

### 3.3 매니폴드 가설 (Manifold Hypothesis)

비선형 차원 축소의 이론적 근거이다. 고차원 데이터는 실제로 고차원 공간 전체에 균일하게 퍼져 있는 것이 아니라, 그보다 훨씬 낮은 차원의 곡면(manifold) 위에 놓여 있다는 가설이다.

예를 들어, 1536차원의 텍스트 임베딩은 1536차원 공간의 극히 일부에만 분포하며, 그 분포는 의미적 관계에 의해 형성된 곡면 위에 놓여 있다. 차원 축소의 목표는 이 곡면을 "펼쳐서" 2차원으로 보여주는 것이다.

비유: 구겨진 종이를 상상해보자. 3차원 공간에서 복잡한 형태를 갖지만, 본질적으로는 2차원 표면이다. 이 종이를 펼치면 2차원에서 원래의 구조를 볼 수 있다. 비선형 차원 축소가 하는 일이 바로 이것이다.

---

## 4. PCA (Principal Component Analysis)

### 4.1 개요

PCA는 가장 오래되고 가장 널리 쓰이는 차원 축소 기법이다. 1901년 Karl Pearson이 제안하고, 1933년 Harold Hotelling이 현대적 형태로 정립했다.

핵심 아이디어: **데이터의 분산(variance)이 가장 큰 방향을 찾아, 그 방향으로 투영한다.** 분산이 큰 방향이 정보가 많은 방향이라는 가정에 기반한다.

### 4.2 수학적 원리

#### 4.2.1 직관적 설명

2차원 데이터가 타원형으로 분포해 있다고 상상해보자. 이 데이터를 1차원(직선)으로 축소해야 한다면, 타원의 긴 축 방향으로 투영하는 것이 정보 손실을 최소화한다. 짧은 축 방향으로 투영하면 점들이 많이 겹치게 되기 때문이다.

PCA는 이 "긴 축 방향"을 수학적으로 찾는 방법이다.

#### 4.2.2 수학적 절차

**Step 1: 데이터 중심화 (Centering)**

N개의 d차원 데이터 벡터 x₁, x₂, ..., xₙ이 있을 때, 평균 벡터를 빼서 원점 중심으로 이동시킨다.

```
μ = (1/N) Σᵢ xᵢ
x̃ᵢ = xᵢ - μ
```

**Step 2: 공분산 행렬 계산 (Covariance Matrix)**

중심화된 데이터로 d×d 크기의 공분산 행렬 C를 계산한다.

```
C = (1/(N-1)) Σᵢ x̃ᵢ x̃ᵢᵀ = (1/(N-1)) X̃ᵀX̃
```

여기서 X̃는 중심화된 데이터 행렬 (N×d)이다. 공분산 행렬의 (i,j) 원소는 i번째 차원과 j번째 차원 사이의 공분산이다.

**Step 3: 고유값 분해 (Eigenvalue Decomposition)**

공분산 행렬 C를 고유값 분해한다.

```
C = VΛVᵀ
```

여기서:
- V는 고유벡터(eigenvector)들을 열로 가지는 직교 행렬 (d×d)
- Λ는 대응하는 고유값(eigenvalue) λ₁ ≥ λ₂ ≥ ... ≥ λ_d를 대각 원소로 가지는 대각 행렬

각 고유벡터 vᵢ는 데이터 분산이 큰 방향(= 주성분, Principal Component)을 나타내고, 대응하는 고유값 λᵢ는 그 방향의 분산 크기를 나타낸다.

**Step 4: 차원 선택 및 투영**

상위 k개의 고유벡터를 선택하여 투영 행렬 W를 구성한다. (d×k)

```
W = [v₁, v₂, ..., vₖ]
```

원본 데이터를 k차원으로 투영한다.

```
yᵢ = Wᵀ x̃ᵢ    (d차원 → k차원)
```

#### 4.2.3 설명 분산 비율 (Explained Variance Ratio)

각 주성분이 전체 분산 중 얼마를 설명하는지 나타내는 비율이다.

```
설명 분산 비율 = λᵢ / Σⱼ λⱼ
```

상위 k개 주성분의 누적 설명 분산 비율이 예를 들어 95%라면, k개 차원만으로 원본 데이터 분산의 95%를 보존한다는 의미이다.

실무에서는 이 누적 비율이 85~95%가 되는 k값을 선택하는 것이 일반적이다.

### 4.3 SVD를 통한 효율적 계산

실제 구현에서는 공분산 행렬을 명시적으로 계산하지 않고, 데이터 행렬 자체에 SVD(Singular Value Decomposition)를 적용한다.

```
X̃ = UΣVᵀ
```

여기서:
- U는 N×N 직교 행렬 (left singular vectors)
- Σ는 N×d 대각 행렬 (singular values σᵢ)
- V는 d×d 직교 행렬 (right singular vectors = 주성분 방향)

공분산 행렬의 고유값과 SVD의 특이값 사이에는 다음 관계가 성립한다:

```
λᵢ = σᵢ² / (N-1)
```

SVD는 공분산 행렬(d×d)을 직접 계산하지 않아도 되므로, d가 매우 클 때(임베딩의 경우 1536 이상) 더 안정적이고 효율적이다.

### 4.4 PCA의 특성 요약

**장점:**
- 계산이 매우 빠르다 (O(min(N²d, Nd²)))
- 결과가 결정적(deterministic)이다 — 같은 데이터에 항상 같은 결과
- `transform()` 메서드로 새 데이터를 즉시 기존 공간에 매핑할 수 있다
- 설명 분산 비율로 정보 손실을 정량화할 수 있다
- 이론적 기반이 견고하고 해석이 명확하다

**단점:**
- 선형 변환만 가능하므로, 비선형 구조를 포착할 수 없다
- 군집(cluster)이 선형적으로 분리되지 않는 경우, 2D 투영에서 군집이 겹쳐 보일 수 있다
- 분산이 큰 방향이 항상 의미 있는 방향은 아닐 수 있다

**임베딩 시각화에서의 역할:**
- 단독으로는 군집 시각화에 부족하지만, UMAP/t-SNE의 전처리 단계로 매우 유용하다
- 1536D → 50D PCA 전처리만으로 UMAP 속도가 3~5배 향상되며, 노이즈 차원 제거 효과로 결과 품질도 개선된다
- "빠른 미리보기" 용도로 활용 가능하다

### 4.5 Python 구현 예시

```python
import numpy as np
from sklearn.decomposition import PCA

# 1536차원 임베딩 10000개
embeddings = np.random.randn(10000, 1536)

# 50차원으로 축소
pca = PCA(n_components=50, random_state=42)
reduced = pca.fit_transform(embeddings)

# 설명 분산 비율 확인
cumulative = np.cumsum(pca.explained_variance_ratio_)
print(f"50차원 누적 설명 분산: {cumulative[-1]:.2%}")
# 예: "50차원 누적 설명 분산: 87.3%"

# 새 데이터 즉시 매핑
new_embedding = np.random.randn(1, 1536)
new_reduced = pca.transform(new_embedding)
```

---

## 5. t-SNE (t-distributed Stochastic Neighbor Embedding)

### 5.1 개요

t-SNE는 2008년 Laurens van der Maaten과 Geoffrey Hinton이 제안한 비선형 차원 축소 알고리즘이다. 이전의 SNE(Stochastic Neighbor Embedding, 2002)를 개선한 것으로, 고차원 데이터의 **지역 구조(이웃 관계)**를 저차원에서 매우 효과적으로 보존한다.

### 5.2 수학적 원리

t-SNE의 핵심 아이디어는 두 단계로 나뉜다:

1. 고차원에서 점들 사이의 "유사도"를 확률 분포로 정의한다.
2. 저차원에서도 점들 사이의 "유사도"를 확률 분포로 정의한다.
3. 두 분포가 최대한 비슷해지도록 저차원 좌표를 최적화한다.

#### 5.2.1 고차원 유사도 (Gaussian 분포)

고차원 공간에서 점 xᵢ가 주어졌을 때, 다른 점 xⱼ가 이웃일 조건부 확률을 가우시안 분포로 정의한다:

```
p(j|i) = exp(-‖xᵢ - xⱼ‖² / 2σᵢ²) / Σₖ≠ᵢ exp(-‖xᵢ - xₖ‖² / 2σᵢ²)
```

여기서 σᵢ는 점 xᵢ 주변의 가우시안 분포의 표준편차이다. σᵢ는 perplexity 파라미터에 의해 결정되며, perplexity는 "유효 이웃 수"를 의미한다. (일반적으로 5~50)

대칭화된 결합 확률:

```
pᵢⱼ = (p(j|i) + p(i|j)) / 2N
```

**직관**: pᵢⱼ가 크다 = 고차원에서 xᵢ와 xⱼ가 가깝다.

#### 5.2.2 저차원 유사도 (t-분포)

저차원 공간의 점 yᵢ, yⱼ 사이의 유사도는 자유도 1의 t-분포(= Cauchy 분포)로 정의한다:

```
qᵢⱼ = (1 + ‖yᵢ - yⱼ‖²)⁻¹ / Σₖ≠ₗ (1 + ‖yₖ - yₗ‖²)⁻¹
```

**왜 가우시안이 아닌 t-분포를 사용하는가?**

이것이 SNE에서 t-SNE로의 핵심 개선이다. t-분포는 가우시안보다 "꼬리가 두꺼운(heavy-tailed)" 분포이다. 이는 저차원에서 중간 거리의 점들이 더 멀어지도록 허용하여, "크라우딩 문제(crowding problem)"를 해결한다.

크라우딩 문제란: 고차원에서 한 점 주변에 많은 이웃이 존재할 수 있지만(차원이 높으므로 공간이 넓다), 이것을 2차원으로 옮기면 모든 이웃을 같은 거리에 배치할 공간이 부족하여 점들이 서로 뭉개져 버리는 현상이다. t-분포를 사용하면 "적당히 먼" 점들이 더 멀어질 수 있어서, 군집 간 분리가 명확해진다.

#### 5.2.3 비용 함수 (KL Divergence)

두 확률 분포 P(고차원)와 Q(저차원)의 차이를 KL 발산(Kullback-Leibler Divergence)으로 측정하고, 이를 최소화하는 저차원 좌표 {yᵢ}를 찾는다:

```
C = KL(P ‖ Q) = Σᵢ Σⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)
```

이 비용 함수의 특성:
- pᵢⱼ가 크고 qᵢⱼ가 작으면 (고차원에서 가까운데 저차원에서 멀면) → 큰 비용
- pᵢⱼ가 작고 qᵢⱼ가 크면 (고차원에서 먼데 저차원에서 가까우면) → 작은 비용

즉, KL 발산은 **가까운 점을 멀리 배치하는 것에 큰 패널티**를 주지만, **먼 점을 가까이 배치하는 것에는 관대**하다. 이것이 t-SNE가 지역 구조는 잘 보존하지만 전역 구조는 약한 근본적인 이유이다.

#### 5.2.4 최적화

경사 하강법(Gradient Descent)으로 비용 함수를 최소화한다.

```
∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ - qᵢⱼ)(yᵢ - yⱼ)(1 + ‖yᵢ - yⱼ‖²)⁻¹
```

초기 좌표는 랜덤으로 설정하며, 보통 1000번 정도 반복한다. 학습률, 모멘텀, early exaggeration 등의 하이퍼파라미터가 결과에 영향을 준다.

### 5.3 주요 하이퍼파라미터

| 파라미터 | 의미 | 일반적 값 | 영향 |
|---|---|---|---|
| perplexity | 유효 이웃 수 | 5~50 (기본 30) | 작으면 미세 구조, 크면 거시 구조 강조 |
| learning_rate | 학습률 | 10~1000 (기본 200) | 너무 작으면 수렴 느림, 너무 크면 불안정 |
| n_iter | 반복 횟수 | 1000~5000 | 충분해야 안정적 결과 |
| early_exaggeration | 초기 확률 증폭 | 12 (기본) | 초기에 군집 형성을 돕는 장치 |

### 5.4 t-SNE의 한계

**군집 간 거리에 의미가 없다**: t-SNE 결과에서 군집 A와 군집 B 사이의 거리가 군집 A와 군집 C 사이의 거리보다 크다고 해서, 실제로 A-B가 A-C보다 더 다르다는 의미가 아니다. 오직 "같은 군집 내의 점들은 실제로 가깝다"는 것만 신뢰할 수 있다.

**군집 크기에 의미가 없다**: 실제로 빽빽한 군집이 t-SNE에서 크게 보이거나, 느슨한 군집이 작게 보일 수 있다. 군집의 크기는 perplexity와 밀도의 상호작용에 의해 결정되며, 원본 데이터의 실제 크기를 반영하지 않는다.

**재현성 부족**: 초기 좌표가 랜덤이고, 비볼록(non-convex) 최적화이므로 실행할 때마다 다른 결과가 나온다. `random_state`를 고정해도, 라이브러리 버전이나 실행 환경에 따라 미세하게 달라질 수 있다.

**새 데이터 추가 불가**: `transform()` 메서드가 없다. 새 데이터가 추가되면 전체를 재계산해야 한다.

**느린 속도**: O(N²) 복잡도. Barnes-Hut 근사를 사용하면 O(N log N)으로 개선되지만, 여전히 UMAP보다 느리다.

### 5.5 Python 구현 예시

```python
from sklearn.manifold import TSNE

# 기본 사용
tsne = TSNE(
    n_components=2,
    perplexity=30,
    metric="cosine",
    learning_rate="auto",
    n_iter=1000,
    random_state=42,
)
coords_2d = tsne.fit_transform(embeddings)

# 대규모 데이터에서는 PCA 전처리 권장
from sklearn.decomposition import PCA

pca_reduced = PCA(n_components=50).fit_transform(embeddings)
coords_2d = TSNE(perplexity=30).fit_transform(pca_reduced)
```

---

## 6. UMAP (Uniform Manifold Approximation and Projection)

### 6.1 개요

UMAP은 2018년 Leland McInnes, John Healy, James Melville이 제안한 알고리즘이다. t-SNE의 시각화 품질을 유지하면서 속도를 대폭 개선하고, 전역 구조도 더 잘 보존한다는 점에서 현재 가장 널리 사용되는 차원 축소 알고리즘이다.

### 6.2 이론적 기반

UMAP의 수학적 기반은 t-SNE와 상당히 다르다. UMAP은 **위상수학(topology)**과 **리만 기하학(Riemannian geometry)**에 기반한다.

핵심 가정: 데이터가 고차원 공간에 균일하게 분포된 리만 다양체(Riemannian manifold) 위에 놓여 있다.

이 가정에서 출발하여 다음을 수행한다:

1. 데이터의 위상적 구조를 "퍼지 단순 복합체(fuzzy simplicial complex)"로 모델링한다.
2. 저차원에서도 동일한 위상 구조를 만들 수 있는 좌표를 찾는다.

#### 6.2.1 고차원 그래프 구성

각 점 xᵢ에 대해 k개의 최근접 이웃을 찾고, 이웃과의 연결 강도를 다음과 같이 정의한다:

```
wᵢⱼ = exp(-(d(xᵢ, xⱼ) - ρᵢ) / σᵢ)
```

여기서:
- d(xᵢ, xⱼ)는 두 점 사이의 거리 (유클리드, 코사인 등)
- ρᵢ는 xᵢ에서 가장 가까운 이웃까지의 거리
- σᵢ는 정규화 상수 (k-nearest neighbor의 거리 분포에 의해 결정)

ρᵢ를 빼는 것이 핵심이다. 이것은 **각 점의 지역 밀도에 맞게 거리를 재조정**하는 효과가 있다. 밀집 지역의 점은 이웃까지의 절대 거리가 짧고, 희소 지역의 점은 이웃까지의 절대 거리가 길다. ρᵢ를 빼면 이 차이가 보정되어, 데이터가 다양체 위에 "균일하게" 분포한다는 가정을 구현한다.

대칭화:

```
wᵢⱼ_sym = wᵢⱼ + wⱼᵢ - wᵢⱼ · wⱼᵢ
```

이 대칭화 방식은 t-SNE의 평균((p(j|i) + p(i|j))/2)과 다르다. UMAP은 "합집합(fuzzy union)" 의미의 대칭화를 사용하여, 한쪽이라도 가깝다고 판단하면 연결을 유지한다.

#### 6.2.2 저차원 유사도

저차원 공간에서의 연결 강도:

```
vᵢⱼ = (1 + a · ‖yᵢ - yⱼ‖²ᵇ)⁻¹
```

여기서 a, b는 `min_dist` 파라미터에 의해 결정되는 상수이다. `min_dist`가 작으면 점들이 더 밀집되고, 크면 더 펼쳐진다.

이 함수의 형태는 t-SNE의 t-분포와 유사하지만, a, b 파라미터로 더 유연하게 조절할 수 있다.

#### 6.2.3 비용 함수 (Cross-Entropy)

UMAP은 KL 발산이 아닌 **교차 엔트로피(Cross-Entropy)**를 사용한다:

```
C = Σᵢⱼ [wᵢⱼ log(wᵢⱼ/vᵢⱼ) + (1-wᵢⱼ) log((1-wᵢⱼ)/(1-vᵢⱼ))]
```

이 비용 함수는 두 개의 항으로 구성된다:
- **인력 항** (첫 번째 항): 고차원에서 가까운 점을 저차원에서도 가깝게 만든다 (t-SNE와 유사)
- **반발 항** (두 번째 항): 고차원에서 먼 점을 저차원에서도 멀게 만든다 (**t-SNE에 없는 것**)

이 반발 항이 UMAP이 전역 구조를 더 잘 보존하는 핵심 이유이다. t-SNE의 KL 발산은 "먼 점을 가까이 배치하는 것에 관대"했지만, UMAP의 교차 엔트로피는 이것에도 패널티를 부과한다.

#### 6.2.4 최적화 (SGD)

UMAP은 확률적 경사 하강법(SGD)을 사용한다. 전체 그래디언트가 아닌, 랜덤 샘플(edge)에 대한 그래디언트로 업데이트하므로 t-SNE보다 훨씬 빠르다.

또한 "음성 샘플링(negative sampling)" 기법을 사용하여, 연결되지 않은 점 쌍(반발 항)을 효율적으로 처리한다. 모든 비연결 쌍을 계산하지 않고, 랜덤으로 몇 개만 골라 반발력을 계산한다.

### 6.3 주요 하이퍼파라미터

| 파라미터 | 의미 | 일반적 값 | 영향 |
|---|---|---|---|
| n_neighbors | 지역 구조 크기 | 5~50 (기본 15) | 작으면 미세 구조, 크면 거시 구조 |
| min_dist | 저차원 최소 거리 | 0.0~0.99 (기본 0.1) | 작으면 밀집, 크면 분산 |
| metric | 거리 함수 | cosine, euclidean 등 | 임베딩에는 cosine 권장 |
| n_components | 출력 차원 | 2 또는 3 | 시각화는 보통 2 |

**n_neighbors의 직관**: "각 점이 세상을 바라보는 시야의 크기." n_neighbors=5면 아주 가까운 이웃만 보고, n_neighbors=50이면 더 넓은 범위의 이웃을 고려한다. 작은 값은 미세한 지역 구조를, 큰 값은 전역적 연결 구조를 강조한다.

**min_dist의 직관**: "축소된 공간에서 점들이 얼마나 가까이 모일 수 있는가." min_dist=0이면 같은 군집의 점들이 거의 겹쳐서 빽빽하게 모이고, min_dist=0.5면 점들 사이에 여유 공간이 생겨 개별 점을 구분하기 쉽다.

### 6.4 UMAP vs. t-SNE 핵심 차이

| 측면 | t-SNE | UMAP |
|---|---|---|
| 이론적 기반 | 확률 분포 매칭 | 위상 공간 보존 |
| 유사도 대칭화 | 평균 | 퍼지 합집합 |
| 비용 함수 | KL 발산 (인력만) | 교차 엔트로피 (인력 + 반발) |
| 최적화 | 배치 GD | SGD + 음성 샘플링 |
| 복잡도 | O(N log N) (Barnes-Hut) | O(N^1.14) (실증적) |
| transform() | 불가 | 가능 |
| 전역 구조 | 약함 | 중~강 |

### 6.5 transform()의 중요성

UMAP은 새 데이터를 기존 학습된 공간에 매핑하는 `transform()` 메서드를 지원한다. 이는 실무에서 매우 중요하다:

- 새 문서가 인덱싱될 때, 전체 UMAP을 재계산하지 않고 기존 좌표계에 추가할 수 있다.
- 검색 쿼리 벡터를 같은 2D 공간에 찍어서, 어느 문서 근처에서 검색이 이루어지는지 시각화할 수 있다.

```python
import umap

# 학습
reducer = umap.UMAP(metric="cosine", random_state=42)
coords_2d = reducer.fit_transform(embeddings)

# 새 데이터 매핑 (전체 재계산 불필요)
new_embeddings = np.random.randn(10, 1536)
new_coords = reducer.transform(new_embeddings)
```

### 6.6 Python 구현 예시

```python
import umap
import numpy as np

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)

# 학습 및 변환
coords_2d = reducer.fit_transform(embeddings)

# 새 데이터 매핑
new_coords = reducer.transform(new_embeddings)
```

---

## 7. TriMap

### 7.1 개요

TriMap은 2019년 Ehsan Amid와 Manfred K. Warmuth가 제안한 알고리즘이다. "triplet" 기반의 접근법으로, 전역 구조와 지역 구조의 균형을 기존 방법들보다 더 잘 달성한다고 주장한다.

### 7.2 핵심 아이디어

t-SNE와 UMAP이 "쌍(pair)" 기반인 반면, TriMap은 **"삼중(triplet)"** 기반이다.

삼중이란: 세 점 (i, j, k)에 대해, "i는 k보다 j에 더 가깝다"는 관계를 의미한다. 이 상대적 거리 순서를 저차원에서도 보존하는 것이 TriMap의 목표이다.

#### 7.2.1 Triplet 선택

각 점 xᵢ에 대해 두 종류의 이웃을 선택한다:
- **Inlier** (가까운 이웃): 해당 점에서 가장 가까운 n_inliers개의 점
- **Outlier** (먼 점): 랜덤하게 선택된 n_outliers개의 점

Triplet (i, j, k)는 "j는 i의 inlier, k는 i의 outlier"로 구성된다.

#### 7.2.2 비용 함수

각 triplet (i, j, k)에 대해:

```
t_ijk = 1 / (1 + ‖yᵢ - yⱼ‖² / ‖yᵢ - yₖ‖²)
```

비용 함수:

```
C = Σ_{(i,j,k)} wᵢⱼₖ · log(1 + t_ijk)
```

여기서 wᵢⱼₖ는 triplet의 가중치로, 고차원에서의 거리 비율에 기반한다.

**핵심 장점**: 이 비용 함수는 절대 거리가 아닌 **상대적 거리 순서**를 보존한다. "A가 B보다 C에 가깝다"는 관계가 유지되면 되므로, 전역 구조(먼 점들 사이의 상대적 배치)도 잘 보존된다.

### 7.3 t-SNE/UMAP과의 비교

| 측면 | t-SNE/UMAP | TriMap |
|---|---|---|
| 기본 단위 | 점 쌍 (pairwise) | 점 삼중 (triplet) |
| 보존 대상 | 절대적 유사도 | 상대적 거리 순서 |
| 전역 구조 | 약~중 | 강 |
| 속도 | UMAP이 빠름 | UMAP과 비슷 |
| transform() | UMAP만 지원 | 미지원 |

### 7.4 하이퍼파라미터

| 파라미터 | 의미 | 기본값 |
|---|---|---|
| n_inliers | 가까운 이웃 수 | 10 |
| n_outliers | 먼 점 수 | 5 |
| n_random | 랜덤 triplet 수 | 5 |
| weight_adj | 가중치 조정 | 500 |

### 7.5 Python 구현 예시

```python
import trimap

reducer = trimap.TRIMAP(
    n_inliers=10,
    n_outliers=5,
    n_random=5,
)
coords_2d = reducer.fit_transform(embeddings)
```

---

## 8. PaCMAP (Pairwise Controlled Manifold Approximation)

### 8.1 개요

PaCMAP은 2021년 Yingfan Wang 등이 제안한 가장 최신의 차원 축소 알고리즘이다. 기존 알고리즘들의 장단점을 분석하고, "세 종류의 점 쌍"을 단계적으로 최적화하는 방식으로 전역-지역 균형을 달성한다.

### 8.2 핵심 아이디어: 세 종류의 점 쌍

PaCMAP은 점 쌍을 세 가지 범주로 나누어 관리한다:

1. **Near Pairs (가까운 쌍)**: k-최근접 이웃 (지역 구조 보존)
2. **Mid-Near Pairs (중간 거리 쌍)**: 중간 거리의 비이웃 점들 (전역 구조 보존의 핵심)
3. **Far Pairs (먼 쌍)**: 랜덤하게 선택된 먼 점들 (반발력)

#### 8.2.1 비용 함수

```
C = w_near · Σ_{(i,j)∈Near} [d²ᵢⱼ / (10 + d²ᵢⱼ)]
  + w_mid  · Σ_{(i,j)∈Mid}  [d²ᵢⱼ / (10000 + d²ᵢⱼ)]
  + w_far  · Σ_{(i,j)∈Far}  [1 / (1 + d²ᵢⱼ)]
```

여기서 dᵢⱼ = ‖yᵢ - yⱼ‖ (저차원 거리)

각 항의 역할:
- **Near 항**: 가까운 점을 가까이 끌어당김 (인력)
- **Mid 항**: 중간 거리 점의 상대적 배치를 보존 (전역 구조의 핵심)
- **Far 항**: 먼 점을 밀어냄 (반발력)

#### 8.2.2 3단계 가중치 스케줄링

PaCMAP의 독특한 특징은 학습 과정에서 세 쌍의 가중치를 **단계적으로 변화**시킨다는 것이다:

| 단계 | 반복 구간 | Near | Mid | Far | 목적 |
|---|---|---|---|---|---|
| Phase 1 | 0~100 | 2 | 1000 | 1 | Mid-near 쌍으로 전역 구조 먼저 잡기 |
| Phase 2 | 100~200 | 3 | 3 | 1 | 균형 잡힌 세밀 조정 |
| Phase 3 | 200~450 | 1 | 0.5 | 0 | Near 쌍으로 지역 구조 다듬기 |

이 스케줄링이 PaCMAP의 핵심 혁신이다.

Phase 1에서 mid-near 쌍의 가중치를 크게 줘서 **먼저 큰 그림(전역 구조)을 잡고**, Phase 3에서 near 쌍에 집중하여 **세부(지역 구조)를 다듬는다**. 이 "거시 → 미시" 전략이 전역-지역 균형을 달성하는 비결이다.

### 8.3 Mid-Near Pairs가 중요한 이유

기존 알고리즘들의 문제를 Mid-Near 관점에서 분석해보면:

- **t-SNE**: 가까운 쌍만 강하게 보존 → 군집은 잘 형성되지만, 군집 간 배치가 제멋대로
- **UMAP**: 가까운 쌍 + 랜덤 반발 → t-SNE보다 낫지만, 중간 거리의 관계가 여전히 약함
- **PaCMAP**: 가까운 + **중간** + 먼 쌍을 모두 명시적으로 관리 → 모든 스케일의 구조가 보존됨

Mid-near 쌍은 "같은 주제이지만 서로 다른 하위 군집에 속하는 점들" 사이의 관계를 보존한다. 이는 계층적 구조를 가진 데이터(예: 같은 카테고리의 다른 문서들)를 시각화할 때 특히 중요하다.

### 8.4 하이퍼파라미터

| 파라미터 | 의미 | 기본값 |
|---|---|---|
| n_neighbors | 가까운 이웃 수 | 10 |
| MN_ratio | mid-near 쌍 비율 | 0.5 |
| FP_ratio | far 쌍 비율 | 2.0 |
| n_components | 출력 차원 | 2 |

PaCMAP의 장점 중 하나는 **하이퍼파라미터에 덜 민감**하다는 것이다. 기본값으로도 대부분의 데이터셋에서 좋은 결과를 보인다. 반면 t-SNE는 perplexity, UMAP은 n_neighbors와 min_dist의 조합에 따라 결과가 크게 달라질 수 있다.

### 8.5 Python 구현 예시

```python
import pacmap

reducer = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=10,
    MN_ratio=0.5,
    FP_ratio=2.0,
)
coords_2d = reducer.fit_transform(embeddings)
```

---

## 9. 종합 비교 분석

### 9.1 정량적 비교

| 기준 | PCA | t-SNE | UMAP | TriMap | PaCMAP |
|---|---|---|---|---|---|
| **속도** (5만 점) | ~1초 | ~2-5분 | ~30초 | ~20초 | ~25초 |
| **전역 구조 보존** | ◎ | △ | ○ | ◎ | ◎ |
| **지역 구조 보존** | △ | ◎ | ◎ | ◎ | ◎ |
| **군집 분리도** | △ | ◎ (과장 경향) | ○ | ○ | ○ |
| **재현성** | ◎ 결정적 | △ 불안정 | ○ seed 고정 | ○ seed 고정 | ○ seed 고정 |
| **새 데이터 추가** | ◎ transform | ✗ | ◎ transform | ✗ | ✗ |
| **하이퍼파라미터 민감도** | 없음 | 높음 | 중간 | 낮음 | 낮음 |
| **이론적 기반** | 선형대수 | 확률론 | 위상수학 | Triplet 기반 | Pairwise 제어 |
| **발표 연도** | 1901/1933 | 2008 | 2018 | 2019 | 2021 |
| **pip 패키지** | scikit-learn | scikit-learn | umap-learn | trimap | pacmap |

### 9.2 전역 vs. 지역 구조 보존의 시각적 차이

동일한 데이터(3개 군집)에 대해 각 알고리즘이 어떻게 다르게 표현하는지를 개념적으로 설명한다.

**PCA:**
```
  군집이 겹쳐 보임
  ┌──────────────┐
  │   ●●●●●●     │
  │  ●●○○○●●●    │    ● ○ ■ = 서로 다른 군집
  │   ○○■■○○     │    군집 경계가 불분명
  │    ■■■■      │
  └──────────────┘
```

**t-SNE:**
```
  군집이 과도하게 분리됨
  ┌──────────────────────┐
  │  ●●●                 │
  │  ●●●    ○○○          │    군집 간 거리는 의미 없음
  │         ○○○          │    각 군집 내부 구조는 잘 보존
  │              ■■■     │
  │              ■■■     │
  └──────────────────────┘
```

**UMAP:**
```
  군집이 분리되고 상대적 거리 일부 보존
  ┌──────────────────────┐
  │  ●●●                 │
  │  ●●●   ○○○           │    군집 A(●)와 B(○)가 가까움
  │        ○○○           │    → 실제로 관련 있는 주제
  │                      │
  │          ■■■         │    군집 C(■)는 떨어져 있음
  │          ■■■         │
  └──────────────────────┘
```

**PaCMAP:**
```
  전역 구조와 지역 구조 모두 잘 보존
  ┌──────────────────────┐
  │  ●●●                 │
  │  ●●●  ○○○            │    군집 간 거리가 의미 있음
  │       ○○○            │    군집 내부 구조도 보존
  │                      │    중간 거리 관계도 반영
  │         ■■■          │
  │         ■■■          │
  └──────────────────────┘
```

### 9.3 비용 함수 비교

각 알고리즘이 "무엇을 최적화하는가"를 비교하면, 결과의 차이를 이해할 수 있다.

| 알고리즘 | 비용 함수 | 인력 | 반발 | 특징 |
|---|---|---|---|---|
| PCA | 재구성 오차 최소화 | - | - | 분산 최대화 (선형) |
| t-SNE | KL(P‖Q) | 강함 | 없음 (암시적) | 가까운 쌍에 편향 |
| UMAP | Cross-Entropy(P,Q) | 강함 | 명시적 | 인력+반발 균형 |
| TriMap | Triplet loss | 상대적 | 상대적 | 순서 보존 |
| PaCMAP | 3-term weighted loss | 단계적 | 단계적 | 거시→미시 스케줄링 |

### 9.4 사용 시나리오별 추천

| 시나리오 | 1순위 추천 | 이유 |
|---|---|---|
| 빠른 미리보기 (탐색적) | PCA | 즉시 결과, 캐싱 불필요 |
| 군집 분리 강조 | t-SNE | 지역 구조 보존 최강 |
| 범용 시각화 | UMAP | 속도·품질·기능 균형 |
| 전역 구조 중시 | PaCMAP 또는 TriMap | 군집 간 상대적 배치가 정확 |
| 실시간 데이터 추가 | UMAP | transform() 지원 |
| 대규모 (10만+) | PCA → UMAP 파이프라인 | 속도와 품질 모두 확보 |
| 하이퍼파라미터 튜닝 최소화 | PaCMAP | 기본값으로 안정적 |

---

## 10. 실무 적용: 2-Stage Pipeline (PCA + UMAP)

### 10.1 왜 2단계 파이프라인인가

단일 UMAP을 1536차원 데이터에 직접 적용할 때의 문제:

1. **속도**: 최근접 이웃 탐색(k-NN)이 고차원에서 매우 느리다. UMAP의 첫 단계인 k-NN 그래프 구성이 전체 시간의 대부분을 차지한다.
2. **노이즈**: 1536차원 중 상당수는 의미 없는 노이즈 차원이다. OpenAI 임베딩의 유효 차원(intrinsic dimensionality)은 50~100 수준이라는 연구 결과가 있다. 노이즈 차원이 포함되면 k-NN 결과도 노이즈에 영향을 받는다.

2-Stage Pipeline은 이 두 문제를 동시에 해결한다:

```
원본 (1536D)  →  PCA (50D)  →  UMAP (2D)
               노이즈 제거     비선형 구조 보존
               속도 향상
```

### 10.2 PCA 중간 차원의 결정

중간 차원은 너무 작으면 정보 손실이 크고, 너무 크면 PCA의 효과가 줄어든다.

#### 경험적 규칙

```
PCA 목표 차원 = max(20, min(100, 원본차원 // 20))
```

| 원본 차원 | 계산 | PCA 목표 |
|---|---|---|
| 384 | 384/20 = 19 → max(20, 19) | 20 |
| 768 | 768/20 = 38 | 38 |
| 1024 | 1024/20 = 51 | 51 |
| 1536 | 1536/20 = 77 | 77 |
| 3072 | 3072/20 = 154 → min(100, 154) | 100 |

#### 설명 분산 기반 결정 (더 정교한 방법)

```python
from sklearn.decomposition import PCA

# 먼저 충분히 큰 차원으로 PCA를 수행하여 설명 분산 확인
pca_full = PCA(n_components=200).fit(embeddings)
cumulative = np.cumsum(pca_full.explained_variance_ratio_)

# 누적 설명 분산이 90%를 넘는 최소 차원을 선택
optimal_dim = np.argmax(cumulative >= 0.90) + 1
print(f"90% 분산 설명에 필요한 차원: {optimal_dim}")
```

### 10.3 성능 벤치마크 (기대값)

| 데이터 규모 | UMAP 직접 (1536D) | PCA(50D) + UMAP | 속도 향상 |
|---|---|---|---|
| 1만 점 | ~30초 | ~8초 | ~3.7x |
| 5만 점 | ~3분 | ~40초 | ~4.5x |
| 10만 점 | ~10분 | ~2분 | ~5x |

실제 속도 향상은 데이터 특성과 하드웨어에 따라 다르지만, 일반적으로 3~5배 향상을 기대할 수 있다.

### 10.4 구현

```python
import numpy as np
from sklearn.decomposition import PCA
import umap

def pca_umap_pipeline(
    embeddings: np.ndarray,
    pca_dim: int = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
):
    """
    2-Stage 차원 축소 파이프라인.

    Parameters:
        embeddings: (N, D) 고차원 임베딩 배열
        pca_dim: PCA 중간 차원 (None이면 자동 결정)
        umap_n_neighbors: UMAP 이웃 수
        umap_min_dist: UMAP 최소 거리
        random_state: 재현성을 위한 시드

    Returns:
        coords_2d: (N, 2) 2D 좌표
        metadata: 파이프라인 메타데이터
    """
    original_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]

    # PCA 중간 차원 결정
    if pca_dim is None:
        pca_dim = max(20, min(100, original_dim // 20))

    metadata = {
        "original_dim": original_dim,
        "n_samples": n_samples,
    }

    # Stage 1: PCA
    if original_dim > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        embeddings_reduced = pca.fit_transform(embeddings)
        metadata["pca_dim"] = pca_dim
        metadata["pca_explained_variance"] = float(
            pca.explained_variance_ratio_.sum()
        )
    else:
        embeddings_reduced = embeddings
        metadata["pca_dim"] = original_dim
        metadata["pca_explained_variance"] = 1.0

    # Stage 2: UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
    )
    coords_2d = reducer.fit_transform(embeddings_reduced)

    metadata["umap_n_neighbors"] = umap_n_neighbors
    metadata["umap_min_dist"] = umap_min_dist

    return coords_2d, metadata
```

---

## 11. 결론 및 알고리즘 선택 가이드

### 11.1 각 알고리즘의 포지셔닝

각 알고리즘은 서로 다른 강점을 가지며, "최고의 알고리즘"은 없다. 사용 목적에 따라 선택해야 한다.

- **PCA**: 가장 빠르고, 이론적으로 가장 투명하다. 전처리 단계나 빠른 탐색에 최적이다. 단독으로는 비선형 구조를 보여주지 못하므로, 최종 시각화용으로는 부족하다.

- **t-SNE**: 군집 분리를 가장 극적으로 보여준다. "이 데이터에 몇 개의 군집이 있는가?"를 파악하기에 좋다. 그러나 군집 간 거리와 크기에 의미가 없으므로, 해석에 주의가 필요하다.

- **UMAP**: 현재 가장 널리 쓰이는 범용 알고리즘이다. 속도, 품질, 기능(`transform()`) 모든 면에서 균형 잡혀 있다. 대부분의 실무 시나리오에서 UMAP이 기본 선택이 된다.

- **TriMap**: 전역 구조 보존에서 UMAP보다 우수하다. 군집 간 상대적 배치가 중요한 분석에 적합하다. 생태계가 작고 `transform()`이 없다는 것이 단점이다.

- **PaCMAP**: 가장 최신이고, 이론적으로 가장 정교한 접근이다. 하이퍼파라미터 튜닝 없이도 안정적인 결과를 보인다. 아직 생태계가 작지만, 점차 채택이 늘고 있다.

### 11.2 실무 의사결정 플로우

```
시작
  │
  ├─ 데이터 규모 10만 이상?
  │    ├─ Yes → PCA 전처리 필수
  │    └─ No  → 직접 적용 가능
  │
  ├─ 새 데이터 실시간 추가 필요?
  │    ├─ Yes → UMAP (transform 지원)
  │    └─ No  → 모든 알고리즘 가능
  │
  ├─ 전역 구조(군집 간 거리) 중요?
  │    ├─ Yes → PaCMAP 또는 TriMap
  │    └─ No  → t-SNE 또는 UMAP
  │
  ├─ 하이퍼파라미터 튜닝 여유 있음?
  │    ├─ Yes → t-SNE (perplexity 조절)
  │    └─ No  → PaCMAP (기본값 안정적)
  │
  └─ 범용적으로 무난한 선택?
       └─ UMAP (PCA 전처리 + UMAP)
```

### 11.3 프로젝트 적용

임베딩 시각화 대시보드 프로젝트에서의 최종 결정:

**기본 파이프라인**: PCA + UMAP
- PCA로 노이즈 제거 및 속도 확보
- UMAP으로 비선형 구조 보존 및 시각화
- `transform()`으로 쿼리 포인트 및 신규 데이터 매핑

**대안 알고리즘 (사용자 선택 옵션)**:
- PCA 단독: 즉시 미리보기용
- PaCMAP: 전역 구조 강조가 필요한 분석
- t-SNE: 군집 분리를 극대화하여 확인하고 싶을 때

---

## 12. 참고문헌

1. Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space." *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science.*

2. Hotelling, H. (1933). "Analysis of a complex of statistical variables into principal components." *Journal of Educational Psychology.*

3. van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." *Journal of Machine Learning Research, 9*, 2579-2605.

4. McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426.*

5. Amid, E., & Warmuth, M. K. (2019). "TriMap: Large-scale Dimensionality Reduction Using Triplets." *arXiv:1910.00204.*

6. Wang, Y., Huang, H., Ruber, C., & Levi, G. (2021). "Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization." *Journal of Machine Learning Research, 22*, 1-73.

7. Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). "On the Surprising Behavior of Distance Metrics in High Dimensional Space." *ICDT.*

8. Kobak, D., & Berens, P. (2019). "The art of using t-SNE for single-cell transcriptomics." *Nature Communications, 10*, 5416.

9. UMAP documentation: https://umap-learn.readthedocs.io/

10. PaCMAP repository: https://github.com/YingfanWang/PaCMAP
