# autoregression 필수 제약 조건인 causal masking을 구현한다.

import numpy as np

seq_len = 8
d_model = 64

# 임베딩 행렬 (seq_len x d_model)
embeddings_matrix = np.random.randn(seq_len, d_model)

# Q, K, V 가중치 행렬
Wq = np.random.randn(d_model, d_model)
Wk = np.random.randn(d_model, d_model)
Wv = np.random.randn(d_model, d_model)

# Q, K, V 계산
Q = embeddings_matrix @ Wq
K = embeddings_matrix @ Wk
V = embeddings_matrix @ Wv

# attention scores (scaled dot-product)
scores = Q @ K.T / np.sqrt(d_model)

# --- causal mask ---
# 상삼각을 -inf로 채워서 미래 토큰을 못 보게 한다
mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
scores[mask] = -np.inf

# softmax (수치 안정성을 위해 max를 빼준다)
shifted = scores - scores.max(axis=-1, keepdims=True)
exp_scores = np.exp(shifted)
attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

# 최종 output
output = attention_weights @ V

print("Scores (masked):")
print(scores)
print("\nAttention weights:")
print(attention_weights)
print("\nOutput shape:", output.shape)