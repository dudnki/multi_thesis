# ======= Text-Prior Logit Fusion (No-CLIP) on Oxford-IIIT Pets =======
# - 이미지: 기존 분류기 특징공간(백본 동결 가능)
# - 텍스트: BERT 라벨-문장 컨텍스트(여러 문장 평균)
# - 정렬: Ridge (BERT -> 이미지 특징공간)
# - 추론: logits_final = logits_base + alpha * prior   (안정화: zero-mean + top-K)
#
# deps: torch, torchvision, transformers, scikit-learn, tqdm, numpy, pandas

import os, json, math, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from my_fuctions.load_text import default_prompts_from_json




# -------------------- Config --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224
BATCH      = 128
NUM_WORKERS= 0          # Windows 안전
EPOCHS_LP  = 5          # 선형 프로브 에폭(백본 고정)
LR_LP      = 1e-3
WEIGHT_DEC = 1e-4
BERT_NAME  = "bert-base-uncased"
RIDGE_LAMBDA   = 1e-2
ALPHAS         = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
TOPK_MASK      = 0    # top-K 재랭킹 (0이면 비활성)
CKPT_PATH      = "classification/save_model/classification_model.pth"   # 기존 모델 체크포인트(.pt) 있으면 경로 지정
TEXT_PATH      = "my_fuctions/text.json"

#-----------------seed----------------------
def set_seed(seed=42):
    import random, os, numpy as np, torch
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Data --------------------
def get_loaders():
    tfm_train = transforms.Compose([
        transforms.Resize(int(IMG_SIZE*1.15), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    tfm_test = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train = OxfordIIITPet("./data", split="trainval", download=True, transform=tfm_train)
    test  = OxfordIIITPet("./data", split="test",     download=True, transform=tfm_test)
    dl_tr = DataLoader(train, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS)
    dl_te = DataLoader(test,  batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    classes = [c.replace("_"," ") for c in train.classes]
    return train, test, dl_tr, dl_te, classes

# -------------------- Model (ResNet, forward_with_features) --------------------
class ResNetFeat(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # avgpool까지
        self.feat_dim = m.fc.in_features
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        # 표준 forward: logits만
        h = self.backbone(x)              # (B, d, 1, 1)
        h = h.flatten(1)                  # (B, d)
        logits = self.head(h)
        return logits

    def forward_with_features(self, x):
        h = self.backbone(x).flatten(1)   # (B, d)
        logits = self.head(h)
        return h, logits

# -------------------- Train (Linear Probe: backbone freeze) --------------------
def train_linear_probe(model, dl_tr, dl_te, class_name, epochs=EPOCHS_LP):
    # 백본 고정
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR_LP, weight_decay=WEIGHT_DEC)
    ce = nn.CrossEntropyLoss()
    model.to(DEVICE).train()
    for ep in range(1, epochs+1):
        losses, accs = [], []
        for x, y in tqdm(dl_tr, desc=f"LP Train ep{ep}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, logits = model.forward_with_features(x)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            pred = logits.argmax(1)
            accs.append((pred==y).float().mean().item()); losses.append(loss.item())
        print(f"[LP] ep{ep}: loss={np.mean(losses):.4f}, acc={np.mean(accs)*100:.2f}%")
    # 검증
    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for x,y in dl_te:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            all_pred.append(logits.argmax(1).cpu()); all_y.append(y.cpu())
    acc = (torch.cat(all_pred)==torch.cat(all_y)).float().mean().item()*100
    print(f"[LP] test acc={acc:.2f}%")
    pred_array = torch.cat(all_pred).cpu().numpy()
    y_array = torch.cat(all_y).cpu().numpy()
    plain_eval = model_eval(y_array, pred_array, class_name)
    torch.save(model.state_dict(), CKPT_PATH)
    return model, plain_eval

# -------------------- Feature & Prototype --------------------
@torch.no_grad()
def extract_feats(model, loader):
    model.eval()
    feats, logits_all, ys = [], [], []
    for x, y in tqdm(loader, desc="Extract feats"):
        x = x.to(DEVICE)
        h, logits = model.forward_with_features(x)
        feats.append(F.normalize(h, dim=-1).cpu())
        logits_all.append(logits.cpu())
        ys.append(y)
    H = torch.cat(feats, 0)       # (N, d)
    L = torch.cat(logits_all, 0)  # (N, K)
    Y = torch.cat(ys, 0)          # (N,)
    return H, L, Y

@torch.no_grad()
def class_image_prototypes(H, Y, num_classes):
    d = H.size(1)
    protos = torch.zeros(num_classes, d)
    counts = torch.zeros(num_classes)
    for k in range(num_classes):
        idx = (Y == k)
        if idx.any():
            m = H[idx].mean(0)
            protos[k] = F.normalize(m, dim=-1)
            counts[k] = idx.sum()
        else:
            protos[k] = torch.zeros(d)
    return protos  # (K, d)


@torch.no_grad()
def build_label_contexts(prompts_dict, layer_idx=8):
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    mdl = AutoModel.from_pretrained(BERT_NAME).to(DEVICE).eval()
    names, vecs = [], []
    for cls, sents in prompts_dict.items():
        sent_vs = []
        for s in sents:
            enc = tok(s, return_tensors="pt").to(DEVICE)
            out = mdl(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx][0]   # (T, d_bert)
            cls_ids = tok(cls, add_special_tokens=False)["input_ids"]
            ids = enc["input_ids"][0].tolist()
            pos = -1
            for i in range(len(ids)-len(cls_ids)+1):
                if ids[i:i+len(cls_ids)] == cls_ids:
                    pos = i; break
            v = hs.mean(0) if pos == -1 else hs[pos:pos+len(cls_ids)].max(0).values
            #v = hs.mean(0) if pos == -1 else hs[pos:pos+len(cls_ids)].mean(0)
            sent_vs.append(v)
        v_cls = torch.stack(sent_vs, 0).max(0).values
        #v_cls = torch.stack(sent_vs, 0).mean(0)
        v_cls = F.normalize(v_cls, dim=-1)
        names.append(cls); vecs.append(v_cls.cpu())
    C = torch.stack(vecs, 0)  # (K, d_bert)
    return names, C

# -------------------- Ridge: BERT -> 이미지 특징공간 --------------------
def fit_align_BERT_to_IMG(C_bert, IMG_protos, ridge_lambda=RIDGE_LAMBDA):
    X = C_bert.numpy()       # (K, d_bert)
    Y = IMG_protos.numpy()   # (K, d_img)
    rows = []
    rg = Ridge(alpha=ridge_lambda, fit_intercept=False)
    for j in range(Y.shape[1]):
        rg.fit(X, Y[:, j])
        rows.append(rg.coef_[None, :])   # (1, d_bert)
    M = np.vstack(rows)                  # (d_img, d_bert)
    return torch.tensor(M, dtype=torch.float32)

# -------------------- Evaluate & Report --------------------
def per_class_report(y_true, y_pred, class_names):
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    rows=[]
    for cls in class_names:
        p, r, f1 = rep[cls]["precision"], rep[cls]["recall"], rep[cls]["f1-score"]
        avg = (p+r+f1)/3
        rows.append((cls, p, r, f1, avg))
    df = pd.DataFrame(rows, columns=["class","precision","recall","f1","avg"]).sort_values("avg")
    return df

# model eval
def model_eval(y_true: np.ndarray, y_pred: np.ndarray, class_names):
    df = per_class_report(y_true, y_pred, class_names)
    print("\n=== classes by (P+R+F1)/3 at best alpha ===")
    #for _, row in df.head(15).iterrows():
    for _, row in df.iterrows():
        print(f"{row['class']:<25} avg={row['avg']*100:5.1f}%  "
            f"(P={row['precision']*100:4.1f}  R={row['recall']*100:4.1f}  F1={row['f1']*100:4.1f})")
    return df


def compare_model(plain_df:pd.DataFrame, method_df:pd.DataFrame, text_class):
    method_df.set_index('class', inplace=True)
    plain_df.set_index('class', inplace=True)
    print("compare       consist text vs plain")
    for class_name in text_class:
        # 'class_name' 인덱스를 사용해 해당 행(Series)을 바로 가져옵니다.
        method_row = method_df.loc[class_name]
        plain_row = plain_df.loc[class_name]
        
        print(f"{class_name:<25} avg={method_row['avg']*100 - plain_row['avg']*100:5.1f}%   "
            f"(P={method_row['precision']*100 - plain_row['precision']*100:4.1f}  "
            f"R={method_row['recall']*100 - plain_row['recall']*100:4.1f}  "
            f"F1={method_row['f1']*100 - plain_row['f1']*100:4.1f})")
        
# ------------------- train_alpha--------------
class LogitFusionModel(nn.Module):
    def __init__(self, initial_alpha=0.0):
        super().__init__()
        # alpha를 학습 가능한 파라미터로 선언합니다.
        # nn.Parameter로 감싸면, PyTorch가 이 텐서를
        # optimizer가 업데이트해야 할 모델의 가중치로 인식합니다.
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))

    def forward(self, logits_base, prior_residual):
        # 최종 로짓을 계산합니다.
        # self.alpha는 학습 과정에서 최적의 값으로 자동 업데이트됩니다.
        final_logits = logits_base + self.alpha * prior_residual
        return final_logits        
        

# -------------------- Main --------------------
def main():
    print(TEXT_PATH)
    set_seed(42)
    # 0) Data
    trainset, testset, dl_tr, dl_te, classes = get_loaders()
    K = len(classes)

    # 1) Model 로드 또는 학습
    model = ResNetFeat(num_classes=K).to(DEVICE)
    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint: {CKPT_PATH}")
        # 체크포인트 로드 시, baseline 성능 평가
        model.eval()
        all_pred, all_y = [], []
        with torch.no_grad():
            for x, y in dl_te:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                all_pred.append(logits.argmax(1).cpu()); all_y.append(y.cpu())
        pred_array = torch.cat(all_pred).cpu().numpy()
        y_array = torch.cat(all_y).cpu().numpy()
        plain_eval = model_eval(y_array, pred_array, classes)
    else:
        print("No checkpoint. Training linear probe (backbone frozen)...")
        model, plain_eval = train_linear_probe(model, dl_tr, dl_te, classes, epochs=EPOCHS_LP)

    # 2) Train과 Test 데이터셋 모두에서 특징과 로짓 추출
    print("Extracting features from training set...")
    H_tr, L_tr, Y_tr = extract_feats(model, dl_tr)
    print("Extracting features from test set...")
    H_te, L_te, Y_te = extract_feats(model, dl_te)

    # 3) 이미지 프로토타입 생성 (학습 데이터 기준)
    IMG_protos = class_image_prototypes(H_tr, Y_tr, num_classes=K)

    # 4) BERT 라벨 컨텍스트 생성
    prompts, in_text_list = default_prompts_from_json(classes, TEXT_PATH)
    names_ctx, C_bert = build_label_contexts(prompts, layer_idx=8)

    # 5) BERT -> 이미지 공간 정렬행렬 학습
    M = fit_align_BERT_to_IMG(C_bert, IMG_protos, ridge_lambda=RIDGE_LAMBDA)
    C_map = F.normalize(C_bert @ M.T, dim=-1)

    # =================================================================
    # ✨ ALPHA 학습을 위한 데이터 준비 (학습 데이터 기준) ✨
    # =================================================================
    
    # (6-train) 학습 데이터에 대한 prior 계산
    prior_tr = (H_tr @ C_map.T)
    
    # (7-train) 학습 데이터에 대한 스케일 보정
    base_zm_tr = L_tr - L_tr.mean(dim=1, keepdim=True)
    prior_zm_tr = prior_tr - prior_tr.mean(dim=1, keepdim=True)
    scale_tr = (base_zm_tr.std() / (prior_zm_tr.std() + 1e-8)).item()
    prior_calib_tr = prior_zm_tr * scale_tr
    
    # (8) LogitFusionModel 학습 시작
    print("\n--- Training learnable alpha ---")
    fusion_model = LogitFusionModel(initial_alpha=0.0).to(DEVICE)
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # DataLoader를 사용하여 미니배치 학습
    from torch.utils.data import TensorDataset
    train_fusion_dataset = TensorDataset(L_tr, prior_calib_tr, Y_tr)
    train_fusion_loader = DataLoader(train_fusion_dataset, batch_size=1024, shuffle=True)
    
    fusion_model.train()
    for epoch in range(200):
        for logits_base_batch, prior_calib_batch, labels_batch in train_fusion_loader:
            final_logits = fusion_model(logits_base_batch.to(DEVICE), prior_calib_batch.to(DEVICE))
            loss = criterion(final_logits, labels_batch.to(DEVICE))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Learned alpha: {fusion_model.alpha.item():.4f}")

    # =================================================================
    # ✨ 최종 평가 (테스트 데이터 기준) ✨
    # =================================================================
    
    # (6-test) 테스트 데이터에 대한 prior 계산
    prior_te = (H_te @ C_map.T)
    
    # (7-test) 테스트 데이터에 대한 스케일 보정
    base_zm_te = L_te - L_te.mean(dim=1, keepdim=True)
    prior_zm_te = prior_te - prior_te.mean(dim=1, keepdim=True)
    scale_te = (base_zm_te.std() / (prior_zm_te.std() + 1e-8)).item()
    prior_calib_te = prior_zm_te * scale_te

    # (9) 학습된 모델로 최종 성능 평가
    fusion_model.eval()
    with torch.no_grad():
        best_alpha_learned = fusion_model.alpha.item()
        print(f"\nTraining finished. Optimal learned alpha = {best_alpha_learned:.4f}")
        
        # 테스트 데이터에 학습된 alpha 적용
        final_logits_test = fusion_model(L_te.to(DEVICE), prior_calib_te.to(DEVICE))
        pred = final_logits_test.argmax(1).cpu().numpy()
        acc = accuracy_score(Y_te.numpy(), pred)
        print(f"Test accuracy with learned alpha: {acc*100:.2f}%")
        
        # 클래스별 성능 리포트
        method_eval = model_eval(Y_te.numpy(), pred, classes)
        compare_model(plain_eval, method_eval, in_text_list) # 필요 시 비교 함수 호출


if __name__ == "__main__":
    main()
