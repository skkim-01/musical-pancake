import numpy as np

# 1일차 np.sqrt(d_k): key dimension으로 일부러 분산을 제한하지 않음
# d_k = K.shape[1]
# scores = Q @ K.T / np.sqrt(d_k)  # shape (5,5)

# 임의의 스코어 벡터
non_scaled_score_1 = np.array([10,10,10,10,10])
non_scaled_score_2 = np.array([1,2,3,4,5])
non_scaled_score_3 = np.array([1,0,-1,-2,3])
non_scaled_score_4 = np.array([1,3,5,7,9])
non_scaled_score_5 = np.array([1,4,8,12,16])

# 루트dk로 나누어 스케일을 하는 경우 분산이 줄어듬을 확인 할 수 있다.
scaled_score_1 = non_scaled_score_1 / np.sqrt(5)
scaled_score_2 = non_scaled_score_2 / np.sqrt(5)
scaled_score_3 = non_scaled_score_3 / np.sqrt(5)
scaled_score_4 = non_scaled_score_4 / np.sqrt(5)
scaled_score_5 = non_scaled_score_5 / np.sqrt(5)

# softmax 함수
# case 4, 5의 출력 결과를 확인해보면 one-hot vector에 가까워짐을 확인 가능하다.
def softmax(x):
    exp_x = np.exp( x- np.max(x, keepdims=True))
    return exp_x / np.sum(exp_x, keepdims=True)


so1 = softmax(non_scaled_score_1)
so2 = softmax(non_scaled_score_2)
so3 = softmax(non_scaled_score_3)
so4 = softmax(non_scaled_score_4)
so5 = softmax(non_scaled_score_5)

print(">>> print non-scaled softmax:")
print(so1)
print(so2)
print(so3)
print(so4)
print(so5)

# 각 case 별로 softmax를 적용한 결과의 합계가 1이 되는 것을 확인한다.
print(">>> print non-scaled sum of softmax:")
print(so1.sum())
print(so2.sum())
print(so3.sum())
print(so4.sum())
print(so5.sum())

# 각 케이스의 스코어가 가장 큰 값에 대한 확률이 1에 가까워지는 것을 확인한다.
print(">>> print max of non-scaled softmax:")
print(so1.max())
print(so2.max())
print(so3.max()) # -> one-hot vector에 가까워짐을 확인 가능하다.
print(so4.max()) # -> one-hot vector에 가까워짐을 확인 가능하다.
print(so5.max()) # -> one-hot vector에 가까워짐을 확인 가능하다.


# dk로 나누어 스케일을 하는 경우 분산이 줄어듬을 확인 할 수 있다.
# 다만 dimension이 작아 one-hot 처럼 보이는 현상(ss5)이 발생한다.
# 실제 트랜스포머의 경우 d_k가 64, 128 등으로 크기 때문에 이러한 현상이 발생하지 않는다.

ss1 = softmax(scaled_score_1)
ss2 = softmax(scaled_score_2)
ss3 = softmax(scaled_score_3)
ss4 = softmax(scaled_score_4)
ss5 = softmax(scaled_score_5)

print(">>> print scaled softmax:")
print(ss1)
print(ss2)
print(ss3)
print(ss4)
print(ss5)

# 각 case 별로 softmax를 적용한 결과의 합계가 1이 되는 것을 확인한다.
print(">>> print scaled sum of softmax:")
print(ss1.sum())
print(ss2.sum())
print(ss3.sum())
print(ss4.sum())
print(ss5.sum())

# 각 케이스의 스코어가 가장 큰 값에 대한 확률이 1에 가까워지는 것을 확인한다.
print(">>> print max of scaled softmax:")
print(ss1.max())
print(ss2.max())
print(ss3.max())
print(ss4.max())
print(ss5.max()) # -> one-hot vector에 가까워짐을 확인 가능하다. 다만 이 현상은 dimension이 작아서 발생하며, 실제로는 dimension이 커질수록 이러한 현상이 발생하지 않는다.
