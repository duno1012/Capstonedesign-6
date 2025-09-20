import librosa
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from sklearn.preprocessing import StandardScaler
import os
import glob
import torch
import torchaudio
import joblib # 모델 저장을 위한 라이브러리

# --- 특징 추출 함수 ---
def get_xvector(file_path, model):
    try:
        signal, fs = torchaudio.load(file_path)
        with torch.no_grad():
            embedding = model.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()
    except Exception as e:
        print(f"'{os.path.basename(file_path)}' 처리 중 오류: {e}")
        return None

# --- 모델 불러오기 ---
print("미리 학습된 x-vector 모델을 불러옵니다...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("모델 로딩 완료!")

# --- 각 가수별 평균 x-vector 생성 및 저장 ---
SINGER_DIRS = ["iu_songs", "younha_songs", "sungsikyung_songs"] # 학습시킬 모든 가수 폴더
print("\n각 가수별 통합 모델 학습 및 저장을 시작합니다...")

for singer_dir in SINGER_DIRS:
    singer_name = os.path.basename(singer_dir).replace("_songs", "")
    print(f"--> '{singer_name}' 학습 중...")
    
    singer_files = glob.glob(os.path.join(singer_dir, '*.wav'))
    all_xvectors = []

    if not singer_files:
        print(f"'{singer_dir}' 폴더에 파일이 없습니다. 건너뜁니다.")
        continue

    for file_path in singer_files:
        xvector = get_xvector(file_path, classifier)
        if xvector is not None:
            all_xvectors.append(xvector)
    
    if all_xvectors:
        # 가수의 평균 x-vector 계산
        singer_avg_xvector = np.mean(all_xvectors, axis=0)
        
        # 학습된 평균 x-vector를 파일로 저장
        joblib.dump(singer_avg_xvector, f'{singer_name}.xvector')
        print(f"'{singer_name}.xvector' 모델 저장 완료!")

print("\n모든 가수 모델의 학습 및 저장이 완료되었습니다.")