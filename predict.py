import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 설정 ──────────────────────────────
MODEL_PATH = "kcelectra_churn_3class_best"  # 모델 폴더 경로
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 128

# ── 모델 로드 ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, num_labels=3).to(DEVICE)
model.eval()

# ── 예측 함수 ──────────────────────────────
def predict_review(text):
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)[0]
        pred   = probs.argmax().item()

    label_map = {0: "이탈없음", 1: "약한불만", 2: "이탈위험"}

    return {
        "label":      label_map[pred],
        "confidence": round(probs[pred].item() * 100, 1),
        "probs": {
            "이탈없음": round(probs[0].item() * 100, 1),
            "약한불만": round(probs[1].item() * 100, 1),
            "이탈위험": round(probs[2].item() * 100, 1),
        }
    }
