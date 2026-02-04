import numpy as np

sample_tokens = ['I', 'had', 'a', 'apple', 'today']

# 임베딩 - token_index를 임베딩한다. tokenizer size to 4차원
np.random.seed(0)
embedding_dim = 4
# 자연어를 학습 가능한 텐서(매트릭스) 형태로 변환 - 트랜스포머 블록 사이즈로 변경
embeddings = np.random.rand(len(sample_tokens), embedding_dim)  # shape (5,4)

print(embeddings)

# 임의의 가중치 값 설정 - 원래대로라면 transformer 가중치 행렬이 지정되어있다.
Wq = np.random.rand(embedding_dim, embedding_dim)
Wk = np.random.rand(embedding_dim, embedding_dim)
Wv = np.random.rand(embedding_dim, embedding_dim)

# @: 행렬 곱셈 연산자 : np.matmul()과 동일 -> shape (5,4)
# Q, K, V는 임베딩된 벡터를 가중치 행렬과 곱해서 얻는다 - 선형 변환 = 순전파
Q = embeddings @ Wq  # = np.matmul(embeddings, Wq)
K = embeddings @ Wk  # = np.matmul(embeddings, Wk)
V = embeddings @ Wv  # = np.matmul(embeddings, Wv)

print(Q)
print(K)
print(V)

# dimension of key | 0: seq_len, 1: embedding_dimension
d_k = K.shape[1]

# Q @ K.T : (5,4) @ (4,5) = (5,5) / np.sqrt(d_k): scaled dot product 
# Q @ K.T 이유는 Q와 K의 유사도를 계산하기 위해서
# d_k로 나누는 이유는 dot product의 값이 너무 커지는 것을 방지
# = softmax 함수 기울기가 0에 가까워지는 것을 방지

# 분산의 성질: 어떤 값 X를 상수 a로 나누면: 분산은 1/a²배가 된다.
# Var(X / a) = Var(X) / a²
# 분산의 정의 : E = Expected Value (기댓값) -> E[X] = 모든 X 값들의 평균
# Var(X) = E[(X - 평균)²]  ← 편차의 "제곱" : 평균에서 얼마나 떨어져 있는지
# 1. (X-평균): 평균에서 얼마나 떨어져 있는지 = 편차
# 2. (X-평균)²: 편차의 "제곱" : 평균에서 얼마나 떨어져 있는지의 제곱 = 음수 제거
# 3. E[(X-평균)²]: 편차의 "제곱"의 평균 : 평균에서 얼마나 떨어져 있는지의 제곱의 평균 = 분산

# 결론은 Q, K의 유사도에 분산의 제한을 두어 softmax 계산 이전 단계인 스코어를 만든다.
# scores.shape = (5,5) = (토큰 수, 토큰 수)
# scores[i][j] = 토큰 i가 토큰 j를 얼마나 참조하는지의 점수
scores = Q @ K.T / np.sqrt(d_k)  # shape (5,5)

# 자연상수 = 미분했을때 자기 자신이 나오는 유일한 지수함수
# f(x) = e^x
# f'(x) = e^x

# 일반 미분 = ln(2) 상수 붙음 = e를 몇제곱 해야 2가 되는지 
# f(x) = 2^x
# f'(x) = 2^x * ln(2)

# 로그와 지수의 관계
# eˣ = y       ↔   ln(y) = x
# e¹ = 2.718   ↔   ln(2.718) = 1
# e⁰ = 1       ↔   ln(1) = 0
# e^0.693 = 2  ↔   ln(2) = 0.693

# 자연상수를 쓰는 이유는 역전파시 미분 계산이 편해진다 - 상수 계산을 하지 않아도 된다.

# softmax function
# 입력값 x(스코어 매트릭스)에 대하여 유사도 점수를 확률로 변환한다.
# x-np.max(x, axis=1, keepdims=True) = x에서 x.shape[1] 의 최대값을 빼서 지수함수 계산시 오버플로우를 방지한다. ex) 스코어가 1000일때 exp(1000)
# e_x = np.exp(x-np.max) = 지수함수를 계산하여 음수를 양수로 변환하며, 역전파 시 미분 계산량을 줄인다 - 자연상수 참조
# e_x / e_x.sum(axis=1) e_x의 값을 e_x의 합계로 나누어 0~1 사이의 확률 값으로 변환한다. 모든 확률의 합은 1이 된다.
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 안정화
    return e_x / e_x.sum(axis=1, keepdims=True)

attention_weights = softmax(scores)  # shape (5,5)
# output = self-attention 결과 = 가중치와 V의 행렬곱으로 얻어지는 결과 = FFN으로 전달함
output = attention_weights @ V  # shape (5,4)

print("Attention Weights:\n", attention_weights)
print("Output Vectors:\n", output)

# 시각화
import matplotlib.pyplot as plt
plt.imshow(attention_weights, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(sample_tokens)), sample_tokens)
plt.yticks(range(len(sample_tokens)), sample_tokens)

plt.savefig('01_qkv_scaled_dot_product.png')