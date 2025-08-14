import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

# ===== 설정 =====
CKPT_PATH = "classification/best_pet_classifier.pth"  # 학습된 모델 경로
ARCH = "resnet50"  # resnet18 / resnet50 / vit_b_16
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
TOPN_ACC = 10   # 정확도 낮은 클래스 개수
TOPN_F1 = 15    # 평균 점수 낮은 클래스 개수

# ===== 데이터 로더 =====
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_dataset = OxfordIIITPet(root="./data", split="test", transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
class_names = [c.replace("_", " ") for c in test_dataset.classes]
num_classes = len(class_names)

# ===== 모델 불러오기 =====
if ARCH == "resnet18":
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif ARCH == "resnet50":
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif ARCH == "vit_b_16":
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
else:
    raise ValueError(f"Unsupported arch: {ARCH}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
model = model.to(device)
model.eval()

# ===== 예측 수집 =====
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===== 클래스별 정확도 =====
accs = []
for c in range(num_classes):
    idx = (y_true == c)
    accs.append((y_pred[idx] == c).mean() if idx.sum() > 0 else 0.0)

df_acc = pd.DataFrame({"class": class_names, "accuracy": accs})
df_acc = df_acc.sort_values("accuracy")

# ===== Classification Report =====
report_dict = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
)

# precision, recall, f1 평균 계산
rows = []
for cls in class_names:
    prec = report_dict[cls]["precision"]
    rec  = report_dict[cls]["recall"]
    f1   = report_dict[cls]["f1-score"]
    avg  = (prec + rec + f1) / 3
    rows.append((cls, prec, rec, f1, avg))

df_scores = pd.DataFrame(rows, columns=["class", "precision", "recall", "f1-score", "avg_score"])
df_scores = df_scores.sort_values("avg_score")

# ===== 출력 =====
print(f"\n전체 정확도: {100 * (y_true == y_pred).mean():.2f}%")

print(f"\n정확도 낮은 클래스 TOP-{TOPN_ACC}:")
for _, row in df_acc.head(TOPN_ACC).iterrows():
    print(f"{row['class']:<25} {row['accuracy']*100:5.1f}%")

print(f"\n평균 점수 낮은 클래스 TOP-{TOPN_F1}:")
for _, row in df_scores.head(TOPN_F1).iterrows():
    print(f"{row['class']:<25} avg={row['avg_score']*100:5.1f}%  "
          f"(P={row['precision']*100:4.1f}  R={row['recall']*100:4.1f}  F1={row['f1-score']*100:4.1f})")
