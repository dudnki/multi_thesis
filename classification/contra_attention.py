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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from my_fuctions.load_text import default_prompts_from_json
from classification.block.attention import AttentionFusionModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset




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
TOPK_MASK      = 10    # top-K 재랭킹 (0이면 비활성)
CKPT_PATH      = "classification/save_model/classification_model.pth"   # 기존 모델 체크포인트(.pt) 있으면 경로 지정
TEXT_PATH      = "my_fuctions/text.json"
EVAL_PATH      = "classification/result/eval.json"

#-----------------seed----------------------
def set_seed(seed=42):
    import random, os, numpy as np, torch
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#-------------------loss---------------------
def info_nce_loss(image_features, text_features, labels, temperature=0.2):
    """
    InfoNCE Loss를 계산하는 함수.
    image_features: (N, D) - 배치 내 이미지 특징
    text_features: (K, D) - 전체 클래스의 텍스트 특징
    labels: (N,) - 배치 내 이미지의 정답 라벨
    """
    # L2 정규화 (코사인 유사도 계산을 위해)
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)

    # 이미지-텍스트 유사도 행렬 계산 (N, K)
    # 각 이미지(행)와 모든 클래스 텍스트(열) 간의 유사도
    logits = image_features @ text_features.T / temperature

    # CrossEntropyLoss는 정답 라벨에 해당하는 logit은 높이고
    # 나머지는 낮추도록 학습합니다. 이것이 대조 학습의 핵심입니다.
    loss = nn.CrossEntropyLoss()
    return loss(logits, labels)

# -------------------- Data --------------------
def get_datasets():
    tfm_train = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)), 
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), # 색상 변환 추가
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
    full_train_dataset  = OxfordIIITPet("./data", split="trainval", download=True, transform=tfm_train)
    test_dataset   = OxfordIIITPet("./data", split="test",     download=True, transform=tfm_test)
    
    train_indices, val_indices = train_test_split(
        list(range(len(full_train_dataset))), 
        test_size=0.1, # 10%를 검증용으로 사용
        random_state=42
    )
    
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)
    
    classes = [c.replace("_", " ") for c in full_train_dataset.classes]
    
    return train_dataset, val_dataset, test_dataset, classes

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
class MLPProjector(nn.Module):
    """
    텍스트 특징 벡터를 이미지 특징 벡터 공간으로 투영(project)하는 MLP.
    """
    def __init__(self, text_dim, image_dim, hidden_dim=512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim)
        )

    def forward(self, text_features):
        return self.projector(text_features)
    
def train_mlp_contrastive(C_bert, H_tr, Y_tr, epochs=100, lr=1e-4):
    """
    전체 학습 데이터를 사용하여 대조 학습(Contrastive Learning) 방식으로 MLP 프로젝터를 학습합니다.
    """
    # 1. 모델 및 옵티마이저 정의
    text_dim = C_bert.shape[1]
    image_dim = H_tr.shape[1]
    
    projector = MLPProjector(text_dim, image_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    
    # 2. 인스턴스 레벨 학습을 위한 데이터 로더 생성
    instance_dataset = TensorDataset(H_tr, Y_tr)
    instance_loader = DataLoader(instance_dataset, batch_size=512, shuffle=True)

    # 3. 학습 루프
    projector.train()
    print("\n--- Training MLP Projector with Contrastive Loss ---")
    
    # C_bert는 고정된 텍스트 프로토타입 셋입니다.
    fixed_C_bert_device = C_bert.to(DEVICE)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for image_features_batch, labels_batch in tqdm(instance_loader, desc=f"Epoch {epoch}"):
            image_features_batch = image_features_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 1. 전체 텍스트 프로토타입을 이미지 공간으로 투영합니다.
            projected_text_protos = projector(fixed_C_bert_device)
            
            # 2. InfoNCE 손실을 계산합니다.
            loss = info_nce_loss(image_features_batch, projected_text_protos, labels_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Avg Contrastive Loss: {total_loss / len(instance_loader):.4f}")

    projector.eval()
    return projector

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
           
        

# -------------------- Main --------------------
def main():
    print(TOPK_MASK)
    set_seed(42)
    
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            try:
                eval_dict = json.load(f)
            except json.JSONDecodeError:
                eval_dict = {}
    else:
        eval_dict = {}
        
    # 0) Data
    train_dataset, val_dataset, test_dataset, classes = get_datasets()
    dl_tr = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    dl_val = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    dl_te = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)
    K = len(classes)

    # 1) Model 로드 또는 학습 (ResNetFeat)
    model = ResNetFeat(num_classes=K).to(DEVICE)
    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint: {CKPT_PATH}")
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
    print("Extracting features from val set...")
    H_val, L_te, Y_val = extract_feats(model, dl_val)
    
    # 3) 이미지 프로토타입 생성 (학습 데이터 기준)
    IMG_protos = class_image_prototypes(H_tr, Y_tr, num_classes=K)

    # 4) BERT 라벨 컨텍스트 생성
    prompts, in_text_list = default_prompts_from_json(classes, TEXT_PATH)
    names_ctx, C_bert = build_label_contexts(prompts, layer_idx=8)

    # 5) BERT -> 이미지 공간 정렬 (대조 학습 기반 MLP 프로젝터)
    # H_tr과 Y_tr 전체를 사용하여 학습합니다.
    projector = train_mlp_contrastive(C_bert, H_tr, Y_tr)

    # 학습된 프로젝터로 C_map 생성 (이 부분은 동일)
    with torch.no_grad():
        C_map = F.normalize(projector(C_bert.to(DEVICE)).cpu(), dim=-1)

    # 학습된 프로젝터로 C_map 생성
    with torch.no_grad():
        # C_bert 전체(37개 클래스)를 DEVICE로 옮겨 추론 후, 결과를 다시 CPU로 가져옴
        C_map = F.normalize(projector(C_bert.to(DEVICE)).cpu(), dim=-1)

    # =================================================================
    # ✨ ATTENTION FUSION MODEL 학습을 위한 데이터 준비 ✨
    # =================================================================
    
    # 이제 AttentionFusionModel은 H_tr(이미지 특징)과 C_map(텍스트 특징)을 직접 입력받습니다.
    # L_tr, prior_calib_tr은 더 이상 직접 사용되지 않습니다.
    
    # num_layer 정의
    num_layer = 2
    
    print(f"\n--- Training AttentionFusionModel_{num_layer}---")
    fusion_model = AttentionFusionModel(
        image_feat_dim=model.feat_dim, # ResNetFeat의 특징 차원
        text_feat_dim=C_map.shape[1],  # C_map의 차원 (d_img와 동일)
        num_classes=K,
        num_layers=num_layer
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4) 
    criterion = nn.CrossEntropyLoss()

    # DataLoader에 이미지 특징(H_tr), 정답 라벨(Y_tr), 텍스트 특징(C_map은 모든 샘플이 동일)
    # 모든 학습 샘플이 동일한 C_map을 참조하므로, DataLoader에 직접 넣지 않고
    # 학습 루프 내부에서 고정된 값으로 전달합니다.
    train_dataset = TensorDataset(H_tr, Y_tr) # 이제 H_tr과 Y_tr만 필요
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True) 
    val_dataset = TensorDataset(H_val, Y_val) # 이제 H_tr과 Y_tr만 필요
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True) 
    
    fixed_C_map_device = C_map.to(DEVICE) # C_map은 한 번만 DEVICE로 옮깁니다.

    best_val_loss = float('inf')
    for epoch in range(1, 101): # 에폭 수 조절
        fusion_model.train()
        train_loss = 0
        for image_features_batch, labels_batch in train_loader:
            image_features_batch = image_features_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            # ❗ AttentionFusionModel에 이미지 특징과 텍스트 특징을 전달
            final_logits = fusion_model(image_features_batch, fixed_C_map_device)
            
            loss = criterion(final_logits, labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        fusion_model.eval()
        val_loss = 0
        with torch.no_grad():

            for image_features_batch, labels_batch in val_loader: 
                image_features_batch = image_features_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                
                final_logits_val = fusion_model(image_features_batch, fixed_C_map_device)
                loss = criterion(final_logits_val, labels_batch)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(dl_val)
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(fusion_model.state_dict(), "classification/save_model/best_fusion_model.pth")
            print(f"  -> Best model saved with val_loss: {best_val_loss:.4f}")
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # =================================================================
    # ✨ 최종 평가 (테스트 데이터 기준) ✨
    # =================================================================
    
    fusion_model.eval()
    with torch.no_grad():
        print(f"\nTraining finished. Evaluating with AttentionFusionModel...")
        
        # 테스트 데이터의 이미지 특징(H_te)과 고정된 텍스트 특징(C_map) 사용
        test_dataset = TensorDataset(H_te, Y_te) # 테스트도 H_te, Y_te만 필요
        test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)
        
        all_pred, all_y = [], []
        for image_features_batch, labels_batch in test_loader:
            image_features_batch = image_features_batch.to(DEVICE)
            
            # ❗ AttentionFusionModel에 테스트 이미지 특징과 텍스트 특징을 전달
            final_logits_test = fusion_model(image_features_batch, fixed_C_map_device)
            
            all_pred.append(final_logits_test.argmax(1).cpu()); all_y.append(labels_batch.cpu())
            
        pred_array = torch.cat(all_pred).cpu().numpy()
        y_array = torch.cat(all_y).cpu().numpy()
        acc = accuracy_score(y_array, pred_array)
        print(f"Test accuracy with AttentionFusionModel: {acc*100:.2f}%")
        
        # 클래스별 성능 리포트
        method_eval = model_eval(y_array, pred_array, classes)
        e_dict_save = method_eval.copy()
        compare_model(plain_eval, method_eval, in_text_list)

        # df를 json으로 변환
        e_dict = e_dict_save.to_dict('records')
        eval_dict['attention_'+str(num_layer)] = e_dict
        with open(EVAL_PATH, "w", encoding="utf-8") as f:
            json.dump(eval_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
