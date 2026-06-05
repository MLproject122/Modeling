# 모델링

## 개요
금융 앱(카카오뱅크, 토스, KB국민은행) 리뷰 텍스트 기반
고객 이탈 위험도 자동 분류 모델

## 분류 기준
| 클래스 | 의미 |
|--------|------|
| 0 | 이탈없음 |
| 1 | 약한불만 |
| 2 | 이탈위험 |

## 모델 성능
| 모델 | Accuracy | Kappa |
|------|----------|-------|
| 로지스틱 회귀 (베이스라인) | 75.2% | 0.640 |
| KcELECTRA (최종) | 85.4% | 0.890 |

## 파일 구조
modeling/
├── baseline_3class.ipynb          # 베이스라인 모델
├── KcELECTRA_3class_pipeline.ipynb # 최종 모델 학습
├── predict.py                      # 예측 함수 (백엔드 연동용)
└── data/
├── train.csv
├── train_3class.csv
├── val_3class.csv
├── test.csv
└── test_3class.csv

## 실행 환경
- Python 3.10
- Google Colab T4 GPU
- transformers, scikit-learn, deep-translator

## 모델 다운로드
[노션 링크 또는 드라이브 링크]

## 사용 방법
```python
from predict import predict_review

result = predict_review("환불 신청했는데 2주째 답장이 없어요")
print(result)
# {"label": "이탈위험", "confidence": 89.1, ...}
```
