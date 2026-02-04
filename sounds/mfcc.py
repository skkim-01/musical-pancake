import librosa

"""
librosa.ex('trumpet')   # 트럼펫
librosa.ex('brahms')    # 브람스 음악
librosa.ex('nutcracker') # 호두까기 인형
librosa.ex('choice')    # 음성
"""

y, sr = librosa.load(librosa.ex('choice'))
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
print(mfcc.shape)