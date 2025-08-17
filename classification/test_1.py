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

# -------------------- Text: BERT 라벨-문장 컨텍스트 평균 --------------------
def default_prompts_for(classes):
    # 클래스명 포함 필수! (라벨 토큰 컨텍스트 사용)
    tmpl = [
        "{c} has distinctive coat patterns.",
        "{c} shows a characteristic ear shape.",
        "{c} typically has specific eye shape and color.",
        "{c} has a typical body size and proportions.",
        "{c} has a recognizable muzzle and face structure.",
        "{c} has a notable tail shape and carriage.",
        "{c} coat length and texture help identify it.",
        #"The facial mask pattern of {c} is a key trait for {c}."
    ]
    d = {c: [t.format(c=c) for t in tmpl] for c in classes}
    # 필요 시 특정 클래스 커스텀 덮어쓰기(예: Siamese 등)
     
    in_text_list = ["American Pit Bull Terrier", "Birman", "Chihuahua", "Staffordshire Bull Terrier",
                    "Wheaten Terrier", "Saint Bernard", "Ragdoll", "Beagle", "English Setter"]
    
    if "American Pit Bull Terrier" in classes:
        d["American Pit Bull Terrier"] = [
            "American Pit Bull Terrier has a muscular and stocky build.",
            "American Pit Bull Terrier has a short coat that is smooth and glossy.",
            "American Pit Bull Terrier shows a broad flat skull and a strong jawline.",
            "American Pit Bull Terrier has ears that may be cropped or natural and are set high.",
            "American Pit Bull Terrier has round to almond-shaped eyes with an alert expression.",
            "American Pit Bull Terrier has a tail that is thick at the base and tapers to a point.",
            "American Pit Bull Terrier appears in many coat colors including brindle fawn and black.",
            "American Pit Bull Terrier stands with a confident and athletic posture."
        ]
    if "Birman" in classes:
        d["Birman"] = [
            "Birman has medium long silky coat with pale body and dark points",
            "Birman has bright blue round eyes",
            "Birman shows white gloves on all four paws a key trait",
            "Birman ears are medium with rounded tips",
            "Birman tail is bushy and proportional",
            "Birman face is sweet and rounded compared to Siamese wedge face",
            "Birman coat is soft and not woolly",
            "Birman body is medium strong not as large as Ragdoll"
        ]
        
    if "Chihuahua" in classes:
        d["Chihuahua"] = [
            "Chihuahua has a compact small frame with a rounded apple head.",
            "Chihuahua has large round expressive eyes.",
            "Chihuahua has large upright ears set at about a 45 degree angle when alert.",
            "Chihuahua has a short slightly pointed muzzle.",
            "Chihuahua may have a smooth coat or a longhaired coat depending on variety.",
            "Chihuahua has a sickle-shaped tail carried over the back.",
            "Chihuahua commonly appears in fawn black chocolate and cream colors.",
            "Chihuahua has a delicate yet well-balanced body."
        ]
    if "Staffordshire Bull Terrier" in classes:
        d["Staffordshire Bull Terrier"] = [
            "Staffordshire Bull Terrier has a broad skull and pronounced cheek muscles.",
            "Staffordshire Bull Terrier has a short smooth close-fitting coat.",
            "Staffordshire Bull Terrier has dark round eyes set wide apart.",
            "Staffordshire Bull Terrier has small ears that are rose-shaped or half-pricked.",
            "Staffordshire Bull Terrier has a wide deep chest.",
            "Staffordshire Bull Terrier has a medium-length tail that tapers to a point.",
            "Staffordshire Bull Terrier appears in brindle red black and white colors.",
            "Staffordshire Bull Terrier has muscular limbs and a sturdy athletic stance."
        ]
    if "Wheaten Terrier" in classes:
        d["Wheaten Terrier"] = [
            "Wheaten Terrier has a soft silky coat with a wheaten pale gold color.",
            "Wheaten Terrier has hair that is wavy or gently flowing and never wiry.",
            "Wheaten Terrier has dark hazel or brown eyes with a friendly look.",
            "Wheaten Terrier has small to medium ears that fold forward.",
            "Wheaten Terrier has a strong square-shaped muzzle.",
            "Wheaten Terrier has a tail carried upright and sometimes docked.",
            "Wheaten Terrier has a coat that often shows a subtle shimmering quality.",
            "Wheaten Terrier has a well-proportioned agile body."
        ]
    if "Saint Bernard" in classes:
        d["Saint Bernard"] = [
            "Saint Bernard has a massive muscular frame.",
            "Saint Bernard has a dense coat that is smooth or slightly rough with white and red-brown markings.",
            "Saint Bernard has a large head with a short muzzle and deep stop.",
            "Saint Bernard has dark brown eyes with a gentle expression.",
            "Saint Bernard has medium-sized ears set high and dropping close to the head.",
            "Saint Bernard has a long heavy slightly curved tail.",
            "Saint Bernard has a broad deep chest.",
            "Saint Bernard often shows a white blaze on the face and white paws."
        ]
    if "Ragdoll" in classes:
        d["Ragdoll"] = [
            "Ragdoll has a semi-long silky plush coat.",
            "Ragdoll has a light-colored body with darker points on the ears face legs and tail.",
            "Ragdoll has striking blue oval-shaped eyes.",
            "Ragdoll has medium-sized ears with rounded tips.",
            "Ragdoll has a large sturdy frame.",
            "Ragdoll has a long bushy tail matching the point color.",
            "Ragdoll has a sweet gentle facial expression.",
            "Ragdoll shows coat patterns such as colorpoint mitted or bicolor."
        ]
    if "Pomeranian" in classes:
        d["Pomeranian"] = [
            "Pomeranian has a dense double coat with a fluffy outer layer.",
            "Pomeranian has a wedge-shaped head with a short muzzle.",
            "Pomeranian has almond-shaped dark eyes.",
            "Pomeranian has small erect ears set high.",
            "Pomeranian has a heavily plumed tail carried over the back.",
            "Pomeranian appears in many coat colors from orange and cream to black and blue.",
            "Pomeranian has a compact squarely built body.",
            "Pomeranian has a distinctive mane of fur around the neck."
        ]
    if "Beagle" in classes:
        d["Beagle"] = [
            "Beagle has a short dense coat with tricolor or bicolor patterns.",
            "Beagle has a slightly domed head with a straight muzzle.",
            "Beagle has brown or hazel eyes with a pleading expression.",
            "Beagle has long low-set ears that hang close to the cheeks.",
            "Beagle has a moderately long tail carried high.",
            "Beagle has a deep chest and a compact body.",
            "Beagle often shows a black saddle with white legs and tan markings.",
            "Beagle has a broad nose with open nostrils."
        ]
    if "English Setter" in classes:
        d["English Setter"] = [
            "English Setter has a long silky coat with feathering on the ears legs and tail.",
            "English Setter shows belton coat patterns in blue orange lemon or liver.",
            "English Setter has a long lean head with a square muzzle.",
            "English Setter has dark brown oval-shaped eyes.",
            "English Setter has low-set ears that hang close to the head.",
            "English Setter has a straight feathered tail carried level with the back.",
            "English Setter has a deep chest with well-sprung ribs.",
            "English Setter moves with a graceful flowing gait."
        ]
    return d, in_text_list

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
    print("\n=== Worst 15 classes by (P+R+F1)/3 at best alpha ===")
    for _, row in df.head(15).iterrows():
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
    set_seed(42)
    # 0) Data
    trainset, testset, dl_tr, dl_te, classes = get_loaders()
    K = len(classes)

    # 1) Model
    model = ResNetFeat(num_classes=K).to(DEVICE)
    if CKPT_PATH and os.path.isfile(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        print(f"Loaded checkpoint: {CKPT_PATH}")
        
        # 1-1) Model eval for per_class
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
        plain_eval = model_eval(y_array, pred_array, classes)
        
    else:
        print("No checkpoint. Training linear probe (backbone frozen)...")
        model, plain_eval = train_linear_probe(model, dl_tr, dl_te, classes, epochs=EPOCHS_LP)
        
    # 2) Train split에서 이미지 특징/프로토타입
    H_tr, L_tr, Y_tr = extract_feats(model, dl_tr)
    IMG_protos = class_image_prototypes(H_tr, Y_tr, num_classes=K)    # (K, d_img)

    # 3) Test split 특징/로짓
    H_te, L_te, Y_te = extract_feats(model, dl_te)                    # H_te normalized
    
    

    # 4) BERT 라벨 컨텍스트
    prompts, in_text_list = default_prompts_for(classes)
    names_ctx, C_bert = build_label_contexts(prompts, layer_idx=8)
    #assert names_ctx == classes, "클래스 순서가 다릅니다."

    # 5) BERT -> 이미지공간 정렬행렬 & 텍스트 프로토타입 매핑
    M = fit_align_BERT_to_IMG(C_bert, IMG_protos, ridge_lambda=RIDGE_LAMBDA)  # (d_img, d_bert)
    C_map = F.normalize(C_bert @ M.T, dim=-1)                                  # (K, d_img)

    # 6) 베이스 로짓(기존 모델) & Prior
    logits_base = L_te.clone()                      # (N, K) 기존 모델 로짓
    prior = (H_te @ C_map.T)                        # (N, K) 텍스트-이미지 유사도
    # 안정화: 제로-평균
    prior = prior - prior.mean(dim=1, keepdim=True)

    # top-K 재랭킹(옵션)
    if TOPK_MASK and TOPK_MASK > 0:
        topk_idx = logits_base.topk(TOPK_MASK, dim=1).indices
        mask = torch.zeros_like(logits_base).scatter(1, topk_idx, 1.0)
    else:
        mask = torch.ones_like(logits_base)
    
    
    # 1) prior와 base의 스케일 확인
    with torch.no_grad():
        base_zm  = logits_base - logits_base.mean(dim=1, keepdim=True)
        prior_zm = prior       - prior.mean(dim=1, keepdim=True)

        print("[scale] base std =", base_zm.std().item(), "prior std =", prior_zm.std().item())

        # 2) top-1 vs top-2 마진과 prior가 뒤집을 수 있는지
        top2_vals, top2_idx = logits_base.topk(2, dim=1)
        margin = (top2_vals[:,0] - top2_vals[:,1]).cpu().numpy()  # 양수일수록 1위가 유리
        print("[margin] mean=", margin.mean(), "p25=", np.percentile(margin,25),
            "p50=", np.percentile(margin,50), "p75=", np.percentile(margin,75))

        # 3) prior가 주는 최대 변화량(행별 max-min)
        delta = (prior * (mask if 'mask' in locals() else 1.0)).cpu().numpy()
        row_span = delta.max(axis=1) - delta.min(axis=1)
        print("[prior span] mean=", row_span.mean(), "p75=", np.percentile(row_span,75))

    scale = (base_zm.std() / (prior_zm.std() + 1e-8)).item()
    prior_calib = prior_zm * scale
    

    # 7) Alpha sweep
    print("\n=== Base vs Ours (Text-Prior Logit Fusion, No-CLIP) ===")
    best_acc, best_a, best_pred = -1, None, None
    for a in ALPHAS:
        logits = logits_base if a==0 else (logits_base + a * prior_calib * mask)
        pred = logits.argmax(1).numpy()
        acc = accuracy_score(Y_te.numpy(), pred)
        print(f"alpha={a:.2f}  Top-1={acc*100:.2f}%")
        if acc > best_acc:
            best_acc, best_a, best_pred = acc, a, pred

    print(f"\nBest alpha: {best_a:.2f}  Top-1={best_acc*100:.2f}%")

    # 8) Worst 15 classes by (P+R+F1)/3 at best alpha
    method_eval = model_eval(Y_te.numpy(), best_pred, classes)

    
    def flip_rate_between(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
        """두 로짓의 argmax가 달라진 비율(예측 뒤집힘 비율)을 반환."""
        pa = logits_a.argmax(dim=1)
        pb = logits_b.argmax(dim=1)
        return (pa != pb).float().mean().item()

    # 스케일링 후 prior 사용
    logits_scaled = logits_base + a * prior_calib * mask
    print("flip_rate vs base =", flip_rate_between(logits_base, logits_scaled))
    
    alphas = np.linspace(0.0, 3.0, 31)  # 0.0 ~ 3.0 사이 31 포인트

    flip_rates = []
    accs = []

    with torch.no_grad():
        for a in alphas:
            logits_new = logits_base + a * prior_calib * mask
            fr = flip_rate_between(logits_base, logits_new)
            pred = logits_new.argmax(dim=1).cpu().numpy()
            acc = accuracy_score(Y_te.numpy(), pred)
            flip_rates.append(fr)
            accs.append(acc)

    # 수치 요약 출력
    print("alpha  |  flip_rate(%)  |  acc(%)")
    for a, fr, acc in zip(alphas, flip_rates, accs):
        print(f"{a:5.2f} | {fr*100:12.3f} | {acc*100:7.3f}")
    
    compare_model(plain_eval, method_eval, in_text_list)

    # (1) Flip rate vs alpha
    plt.figure()
    plt.plot(alphas, flip_rates, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("flip rate")
    plt.title("Flip rate vs alpha")
    plt.grid(True)
    plt.show()

    # (2) Accuracy vs alpha (원한다면 함께 확인)
    plt.figure()
    plt.plot(alphas, accs, marker='o')
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs alpha")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
