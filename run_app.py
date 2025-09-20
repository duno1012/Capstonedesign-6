import os
import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
import torchaudio
import glob
import joblib # 모델 불러오기를 위한 라이브러리

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

# --- 1. 저장된 모델들 불러오기 ---
print("미리 학습된 x-vector 모델을 불러옵니다...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
print("모델 로딩 완료!")

print("\n저장된 가수 모델들을 불러옵니다...")
singer_models = {}
for model_file in glob.glob('*.xvector'):
    singer_name = os.path.splitext(model_file)[0]
    singer_models[singer_name] = joblib.load(model_file)
print(f"--> {list(singer_models.keys())} 모델 로딩 완료!")

# --- 2. 사용자 목소리 분석 ---
USER_VOICE_PATH = "younha_songs/윤하 - 사건의 지평선 MR제거.wav" # <-- 분석하고 싶은 사용자 파일
print(f"\n사용자 목소리 '{USER_VOICE_PATH}'를 분석합니다...")
user_xvector = get_xvector(USER_VOICE_PATH, classifier)

if user_xvector is not None:
    # --- 3. 모든 가수 모델과 유사도 비교 ---
    results = {}
    for singer_name, singer_avg_xvector in singer_models.items():
        similarity = cosine_similarity(singer_avg_xvector.reshape(1, -1), user_xvector.reshape(1, -1))
        results[singer_name] = similarity[0][0] * 100
        
    # --- 4. 최종 결과 발표 ---
    if results:
        best_match_singer = max(results, key=results.get)
        
        print("\n--- 최종 분석 결과 ---")
        # 점수가 높은 순으로 정렬하여 출력
        for singer, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
            print(f"'{singer}' 모델과의 유사도: {score:.2f}%")
            
        print(f"\n==> 최종 판정: 가장 유사한 가수는 '{best_match_singer}' 입니다!")
    else:
        print("결과를 계산할 수 없습니다.")