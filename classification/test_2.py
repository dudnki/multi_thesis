# ======= Text-Prior Logit Fusion (No-CLIP) on Oxford-IIIT Pets =======
# - 이미지: 기존 분류기 특징공간(ResNet18, 백본 동결 후 선형프로브)
# - 텍스트: BERT 라벨-문장 컨텍스트(여러 문장 평균) + 다중 레이어 앙상블
# - 정렬: Ridge (BERT -> 이미지 특징공간)  [타깃: 정규화 전 클래스 평균]
# - 추론: logits_final = logits_base + alpha * prior_residual   (zero-mean, 잔차화, 분산정합, top-K mask)
#
# deps: torch, torchvision, transformers, scikit-learn, tqdm, numpy, pandas, matplotlib(옵션)

import os, random, json, math, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Ridge
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# -------------------- Config --------------------
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE      = 224
BATCH         = 128
NUM_WORKERS   = 0          # Windows 안전
EPOCHS_LP     = 5          # 선형 프로브 에폭(백본 고정)
LR_LP         = 1e-3
WEIGHT_DEC    = 1e-4
BERT_NAME     = "bert-base-uncased"
RIDGE_LAMBDA  = 1e-2
ALPHAS        = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
TOPK_MASK     = 5          # top-K 재랭킹 (0이면 비활성, len(classes)면 해제)
CKPT_PATH     = "classification/save_model/classification_model.pth"       # 선형프로브 가중치 저장/불러오기 경로 (.pt)
SEED          = 42

# ---- Text context 강화: 여러 BERT hidden layers 앙상블 ----
CONTEXT_LAYERS = [6, 8, 10]   # 예: [6,8,10] 또는 [4,8,12]

# ---- 진단/플롯 옵션 ----
PRINT_DIAG_SCALE = True       # base/prior 표준편차, margin, span 프린트
PLOT_CURVES      = False      # flip rate/accuracy vs alpha 플롯 (matplotlib 필요)

# -------------------- Seed --------------------
def set_seed(seed=42):
    import numpy as _np, torch as _tc
    random.seed(seed); _np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    _tc.manual_seed(seed); _tc.cuda.manual_seed_all(seed)
    _tc.backends.cudnn.deterministic = True
    _tc.backends.cudnn.benchmark = False

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

# -------------------- Model --------------------
class ResNetFeat(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # avgpool까지
        self.feat_dim = m.fc.in_features
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        h = self.backbone(x).flatten(1)     # (B, d)
        logits = self.head(h)
        return logits

    def forward_with_features(self, x):
        h = self.backbone(x).flatten(1)     # (B, d)
        logits = self.head(h)
        return h, logits

# -------------------- Train (Linear Probe: backbone freeze) --------------------
def train_linear_probe(model, dl_tr, dl_te, class_name, epochs=EPOCHS_LP):
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

    # test quick check
    model.eval()
    all_pred, all_y = [], []
    with torch.no_grad():
        for x,y in dl_te:
            x = x.to(DEVICE)
            logits = model(x)
            all_pred.append(logits.argmax(1).cpu()); all_y.append(y.cpu())
    acc = (torch.cat(all_pred)==torch.cat(all_y)).float().mean().item()*100
    print(f"[LP] test acc={acc:.2f}%")
    pred_array = torch.cat(all_pred).cpu().numpy()
    y_array = torch.cat(all_y).cpu().numpy()
    plane_model_eval(y_array, pred_array, class_name)
    torch.save(model.state_dict(), CKPT_PATH)
    return model

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
    H = torch.cat(feats, 0)       # (N, d)   normalized
    L = torch.cat(logits_all, 0)  # (N, K)
    Y = torch.cat(ys, 0)          # (N,)
    return H, L, Y

@torch.no_grad()
def class_image_prototypes(H, Y, num_classes, return_raw=False):
    """
    클래스별 이미지 특징 평균.
    - Ridge 타깃은 정규화 전 평균(protos_raw)을 사용.
    - 점수계산용으로는 정규화된 평균(protos_norm)도 반환 가능.
    """
    d = H.size(1)
    protos_raw = torch.zeros(num_classes, d)
    counts = torch.zeros(num_classes)
    for k in range(num_classes):
        idx = (Y == k)
        if idx.any():
            # 주의: H는 normalize된 특징. raw mean이지만 length는 ~1 부근.
            m = H[idx].mean(0)
            protos_raw[k] = m
            counts[k] = idx.sum()
        else:
            protos_raw[k] = torch.zeros(d)
    protos_norm = F.normalize(protos_raw, dim=-1)
    return (protos_norm, protos_raw) if return_raw else protos_norm



def inject_discriminative_prompts(d):
    upd = {
        "American Pit Bull Terrier": [
            "American Pit Bull Terrier has a muscular lean body taller and longer than Staffordshire Bull Terrier",
            "American Pit Bull Terrier shows a broad flat skull and powerful jaw with a defined stop",
            "American Pit Bull Terrier has a short glossy coat and tight skin",
            "American Pit Bull Terrier often has cropped or high set ears giving a taller outline",
            "American Pit Bull Terrier chest is deep but not overly wide compared to Staffordshire Bull Terrier",
            "American Pit Bull Terrier legs are longer and straighter giving an athletic stance",
            "American Pit Bull Terrier tail is thick at base and tapers carried straight",
            "American Pit Bull Terrier expression is alert and confident"
        ],
        "Staffordshire Bull Terrier": [
            "Staffordshire Bull Terrier has a compact stocky body shorter and thicker than American Pit Bull Terrier",
            "Staffordshire Bull Terrier shows a very broad head and pronounced cheek muscles",
            "Staffordshire Bull Terrier has a short close coat and wide front",
            "Staffordshire Bull Terrier chest is wide and deep with strong forequarters",
            "Staffordshire Bull Terrier legs are shorter with heavy bone and a low center of gravity",
            "Staffordshire Bull Terrier ears are small rose or half pricked",
            "Staffordshire Bull Terrier tail is medium length and tapers carried low",
            "Staffordshire Bull Terrier stance appears sturdy and powerful"
        ],
        "Ragdoll": [
            "Ragdoll has semi long silky coat with colorpoint pattern and blue eyes",
            "Ragdoll body is large and heavy with a relaxed gentle posture",
            "Ragdoll ears are medium with rounded tips and set slightly forward",
            "Ragdoll tail is long and bushy matching point color",
            "Ragdoll face is sweet with a flat plane between the eyes",
            "Ragdoll coat shows colorpoint mitted or bicolor patterns",
            "Ragdoll has a light body color with darker extremities",
            "Ragdoll expression is soft and calm compared to Siamese which is slimmer"
        ],
        "Bengal": [
            "Bengal has a sleek muscular body with wild appearance",
            "Bengal coat shows rosettes or large spots unlike Egyptian Mau small spots",
            "Bengal pelt is short dense and often has glitter",
            "Bengal head is small relative to body with prominent whisker pads",
            "Bengal tail is thick with rounded tip often ringed",
            "Bengal moves with athletic gait and strong hindquarters",
            "Bengal ears are medium small and rounded tips",
            "Bengal eyes are oval to round often green or gold"
        ],
        "Maine Coon": [
            "Maine Coon is very large with rectangular body and strong bone",
            "Maine Coon shows tufted lynx like ears and long shaggy coat",
            "Maine Coon tail is very long and bushy",
            "Maine Coon muzzle is square unlike British Shorthair round muzzle",
            "Maine Coon has a ruff around the neck and breeches on hind legs",
            "Maine Coon paws are large often with toe tufts",
            "Maine Coon head is slightly longer than wide",
            "Maine Coon expression is alert and friendly"
        ],
        "Birman": [
            "Birman has medium long silky coat with pale body and dark points",
            "Birman has bright blue round eyes",
            "Birman shows white gloves on all four paws a key trait",
            "Birman ears are medium with rounded tips",
            "Birman tail is bushy and proportional",
            "Birman face is sweet and rounded compared to Siamese wedge face",
            "Birman coat is soft and not woolly",
            "Birman body is medium strong not as large as Ragdoll"
        ],
        "British Shorthair": [
            "British Shorthair has a round face and chubby cheeks",
            "British Shorthair coat is dense plush and crisp",
            "British Shorthair body is cobby with short thick legs",
            "British Shorthair eyes are large round copper or gold",
            "British Shorthair muzzle is short and round unlike Maine Coon square muzzle",
            "British Shorthair ears are small and widely set",
            "British Shorthair tail is thick with a rounded tip",
            "British Shorthair expression is calm and teddy bear like"
        ],
        "American Bulldog": [
            "American Bulldog has a large athletic body taller than Boxer",
            "American Bulldog head is broad with strong muzzle and undershot bite in some lines",
            "American Bulldog chest is deep with muscular shoulders",
            "American Bulldog coat is short and close often white with patches",
            "American Bulldog ears can be rose or semi erect",
            "American Bulldog tail is thick at base and tapers",
            "American Bulldog stance is powerful yet agile",
            "American Bulldog movement shows strong drive from rear"
        ],
        "Russian Blue": [
            "Russian Blue has short dense bluish gray coat with silver sheen",
            "Russian Blue eyes are vivid green a key trait",
            "Russian Blue body is fine boned and graceful",
            "Russian Blue head has a smooth wedge with large ears",
            "Russian Blue tail is long and tapering",
            "Russian Blue whisker pads are prominent giving a gentle smile",
            "Russian Blue coat is double and plush to touch",
            "Russian Blue expression is reserved and elegant"
        ],
        "Abyssinian": [
            "Abyssinian has a slender athletic body with ticked coat",
            "Abyssinian coat shows agouti ticking not spots or stripes",
            "Abyssinian ears are large and alert",
            "Abyssinian eyes are almond shaped gold to green",
            "Abyssinian tail is long and tapering",
            "Abyssinian head is slightly rounded wedge",
            "Abyssinian coat color is warm ruddy or sorrel tones",
            "Abyssinian stance looks lively and curious"
        ],
        "Chihuahua": [
            "Chihuahua has a tiny compact body with apple shaped head",
            "Chihuahua eyes are large round and expressive",
            "Chihuahua ears are very large and upright",
            "Chihuahua muzzle is short and slightly pointed",
            "Chihuahua tail is sickle shaped and carried over the back",
            "Chihuahua coat can be smooth or long",
            "Chihuahua legs are fine and delicate",
            "Chihuahua expression is alert and lively"
        ],
        "Persian": [
            "Persian has a long flowing coat and heavy bone",
            "Persian face is flat with a short nose and large round eyes",
            "Persian body is cobby with low legs",
            "Persian ears are small and rounded set low",
            "Persian tail is short and plumed",
            "Persian coat requires full grooming and appears full around neck",
            "Persian eyes are copper or blue or odd eyes depending on color",
            "Persian expression is sweet and serene"
        ],
        "Boxer": [
            "Boxer has a medium large muscular body with deep chest",
            "Boxer head shows a distinct stop and strong muzzle with undershot bite",
            "Boxer coat is short and tight often fawn or brindle with white markings",
            "Boxer ears may be natural or cropped and are set high",
            "Boxer tail is set high and often docked in some regions",
            "Boxer movement is energetic with springy gait",
            "Boxer expression is alert and playful",
            "Boxer outline is more athletic and less bulky than American Bulldog"
        ],
        "Egyptian Mau": [
            "Egyptian Mau has a naturally spotted coat with small round spots",
            "Egyptian Mau body is medium and graceful with longer hind legs",
            "Egyptian Mau eyes are gooseberry green a key trait",
            "Egyptian Mau head is slightly rounded wedge with visible mascara lines",
            "Egyptian Mau tail is medium long with dark rings",
            "Egyptian Mau coat shows spots on body and stripes on legs and tail",
            "Egyptian Mau ears are medium large and alert",
            "Egyptian Mau expression is vivid and watchful"
        ],
        "Miniature Pinscher": [
            "Miniature Pinscher has a small compact square body",
            "Miniature Pinscher coat is smooth short and glossy",
            "Miniature Pinscher head is wedge shaped with defined stop",
            "Miniature Pinscher eyes are dark oval and keen",
            "Miniature Pinscher ears are high set and often erect",
            "Miniature Pinscher tail is high set and straight",
            "Miniature Pinscher common colors are black and tan solid red and stag red",
            "Miniature Pinscher moves with a high stepping gait"
        ]
    }
    for k, v in upd.items():
        if k in d:
            d[k] = v
    return d



# -------------------- Text: prompts --------------------
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
    ]
    d = {c: [t.format(c=c) for t in tmpl] for c in classes}

    d = inject_discriminative_prompts(d)
    return d

# -------------------- Text: BERT 컨텍스트 (다중 레이어 앙상블) --------------------
@torch.no_grad()
def build_label_contexts_multi(prompts_dict, layers=CONTEXT_LAYERS):
    """
    여러 hidden layer의 라벨-토큰 컨텍스트를 추출해 평균(앙상블)합니다.
    - 각 문장: 라벨 토큰 위치 평균 (없으면 문장 평균) -> 문장 벡터
    - 클래스: 문장 벡터 평균 -> 클래스 벡터
    - 레이어: 클래스 벡터들을 레이어별로 구해 평균
    """
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    mdl = AutoModel.from_pretrained(BERT_NAME).to(DEVICE).eval()
    names = list(prompts_dict.keys())
    cls_vecs_all_layers = []

    for layer_idx in layers:
        vecs = []
        for cls in names:
            sents = prompts_dict[cls]
            sent_vs = []
            for s in sents:
                enc = tok(s, return_tensors="pt").to(DEVICE)
                out = mdl(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer_idx][0]   # (T, d_bert)
                # 라벨 토큰 span 찾기
                cls_ids = tok(cls, add_special_tokens=False)["input_ids"]
                ids = enc["input_ids"][0].tolist()
                pos = -1
                for i in range(len(ids)-len(cls_ids)+1):
                    if ids[i:i+len(cls_ids)] == cls_ids:
                        pos = i; break
                v = hs.mean(0) if pos == -1 else hs[pos:pos+len(cls_ids)].mean(0)
                sent_vs.append(v)
            v_cls = torch.stack(sent_vs, 0).mean(0)    # 문장 평균
            v_cls = F.normalize(v_cls, dim=-1)         # 레이어별 정규화
            vecs.append(v_cls.cpu())
        C_layer = torch.stack(vecs, 0)                 # (K, d_bert)
        cls_vecs_all_layers.append(C_layer)

    # 레이어 앙상블 평균 후 정규화
    C_bert = torch.stack(cls_vecs_all_layers, 0).mean(0)
    C_bert = F.normalize(C_bert, dim=-1)
    return names, C_bert

# -------------------- Ridge: BERT -> 이미지 특징공간 --------------------
def fit_align_BERT_to_IMG(C_bert, IMG_targets_raw, ridge_lambda=RIDGE_LAMBDA):
    """
    minimize || C_bert M^T - IMG_targets_raw ||_F^2 + λ ||M||_F^2
    - X = C_bert (K, d_bert),  Y = IMG_targets_raw (K, d_img)
    - 반환 M shape: (d_img, d_bert)
    """
    X = C_bert.numpy()
    Y = IMG_targets_raw.numpy()
    rows = []
    rg = Ridge(alpha=ridge_lambda, fit_intercept=False)
    for j in range(Y.shape[1]):
        rg.fit(X, Y[:, j])
        rows.append(rg.coef_[None, :])   # (1, d_bert)
    M = np.vstack(rows)                  # (d_img, d_bert)
    return torch.tensor(M, dtype=torch.float32)

# -------------------- Report --------------------
def per_class_report(y_true, y_pred, class_names):
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    rows=[]
    for cls in class_names:
        p, r, f1 = rep[cls]["precision"], rep[cls]["recall"], rep[cls]["f1-score"]
        avg = (p+r+f1)/3
        rows.append((cls, p, r, f1, avg))
    df = pd.DataFrame(rows, columns=["class","precision","recall","f1","avg"]).sort_values("avg")
    return df

def print_worst_15(df):
    print("\n=== Worst 15 classes by (P+R+F1)/3 ===")
    for _, row in df.head(15).iterrows():
        print(f"{row['class']:<25} avg={row['avg']*100:5.1f}%  "
              f"(P={row['precision']*100:4.1f}  R={row['recall']*100:4.1f}  F1={row['f1']*100:4.1f})")

def plane_model_eval(y_true: np.ndarray, y_pred: np.ndarray, class_names):
    df = per_class_report(y_true, y_pred, class_names)
    print("\n=== Worst 15 classes by (P+R+F1)/3 at best alpha ===")
    for _, row in df.head(15).iterrows():
        print(f"{row['class']:<25} avg={row['avg']*100:5.1f}%  "
            f"(P={row['precision']*100:4.1f}  R={row['recall']*100:4.1f}  F1={row['f1']*100:4.1f})")

# -------------------- Main --------------------
def main():
    set_seed(SEED)

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
        plane_model_eval(y_array, pred_array, classes)
    else:
        print("No checkpoint. Training linear probe (backbone frozen)...")
        model = train_linear_probe(model, dl_tr, dl_te,classes, epochs=EPOCHS_LP)
        if CKPT_PATH:
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"Saved checkpoint: {CKPT_PATH}")

    # 2) Train split 이미지 특징/프로토타입 (raw + norm)
    H_tr, L_tr, Y_tr = extract_feats(model, dl_tr)
    IMG_protos_norm, IMG_protos_raw = class_image_prototypes(H_tr, Y_tr, num_classes=K, return_raw=True)

    # 3) Test split 특징/로짓
    H_te, L_te, Y_te = extract_feats(model, dl_te)

    # 4) BERT 라벨 컨텍스트 (다중 레이어 앙상블)
    prompts = default_prompts_for(classes)
    names_ctx, C_bert = build_label_contexts_multi(prompts, layers=CONTEXT_LAYERS)
    assert names_ctx == classes, "클래스 순서가 다릅니다."

    # 5) BERT -> 이미지공간 정렬행렬 (Ridge 타깃: 정규화 전 평균)
    M = fit_align_BERT_to_IMG(C_bert, IMG_protos_raw, ridge_lambda=RIDGE_LAMBDA)  # (d_img, d_bert)
    C_map = F.normalize(C_bert @ M.T, dim=-1)                                     # (K, d_img)

    # 6) 베이스 로짓 & prior 계산
    logits_base = L_te.clone()                      # (N, K)
    prior_raw   = (H_te @ C_map.T)                  # (N, K)

    # ---- (1) 잔차화 prior + 분산 정합 ----
    base_zm  = logits_base - logits_base.mean(dim=1, keepdim=True)
    prior_zm = prior_raw   - prior_raw.mean(dim=1, keepdim=True)

    # prior를 base에 대해 최소제곱 잔차화(중복 성분 제거)
    beta = (base_zm * prior_zm).sum(dim=1, keepdim=True) / (base_zm.pow(2).sum(dim=1, keepdim=True) + 1e-8)
    prior_res = prior_zm - beta * base_zm

    # 분산 정합(variance match)
    scale = (base_zm.std() / (prior_res.std() + 1e-8)).item()
    prior_res = prior_res * scale

    # 안정화: top-K 재랭킹(옵션)
    if TOPK_MASK and TOPK_MASK > 0:
        topk_idx = logits_base.topk(TOPK_MASK, dim=1).indices
        mask = torch.zeros_like(logits_base).scatter(1, topk_idx, 1.0)
    else:
        mask = torch.ones_like(logits_base)

    # 디버그 스케일/마진 출력
    if PRINT_DIAG_SCALE:
        with torch.no_grad():
            base_std  = (base_zm.std().item())
            prior_std = (prior_res.std().item())
            print(f"[scale after residualization] base std = {base_std:.6f}  prior_res std = {prior_std:.6f}")
            top2_vals, _ = logits_base.topk(2, dim=1)
            margin = (top2_vals[:,0] - top2_vals[:,1]).cpu().numpy()
            print("[margin] mean=", margin.mean(), "p25=", np.percentile(margin,25),
                  "p50=", np.percentile(margin,50), "p75=", np.percentile(margin,75))
            delta = (prior_res * mask).cpu().numpy()
            row_span = delta.max(axis=1) - delta.min(axis=1)
            print("[prior_res span] mean=", row_span.mean(), "p75=", np.percentile(row_span,75))

    # 7) Alpha sweep (잔차화 prior 사용)
    print("\n=== Base vs Ours (Text-Prior Logit Fusion, residualized) ===")
    best_acc, best_a, best_pred = -1, None, None
    for a in ALPHAS:
        logits = logits_base if a==0 else (logits_base + a * prior_res * mask)
        pred = logits.argmax(1).numpy()
        acc = accuracy_score(Y_te.numpy(), pred)
        print(f"alpha={a:.2f}  Top-1={acc*100:.2f}%")
        if acc > best_acc:
            best_acc, best_a, best_pred = acc, a, pred

    print(f"\nBest alpha: {best_a:.2f}  Top-1={best_acc*100:.2f}%")

    # 8) Per-class report
    df = per_class_report(Y_te.numpy(), best_pred, classes)
    print_worst_15(df)

    # (옵션) 플롯
    if PLOT_CURVES:
        import matplotlib.pyplot as plt
        alphas = np.linspace(0.0, 3.0, 31)
        flip_rates, accs = [], []
        with torch.no_grad():
            for a in alphas:
                logits_new = logits_base + a * prior_res * mask
                pa = logits_base.argmax(1)
                pb = logits_new.argmax(1)
                fr = (pa != pb).float().mean().item()
                acc = accuracy_score(Y_te.numpy(), pb.cpu().numpy())
                flip_rates.append(fr)
                accs.append(acc)
        print("alpha  |  flip_rate(%)  |  acc(%)")
        for a, fr, acc in zip(alphas, flip_rates, accs):
            print(f"{a:5.2f} | {fr*100:12.3f} | {acc*100:7.3f}")
        plt.figure(); plt.plot(alphas, flip_rates, marker='o'); plt.xlabel("alpha"); plt.ylabel("flip rate"); plt.title("Flip rate vs alpha"); plt.grid(True); plt.show()
        plt.figure(); plt.plot(alphas, accs, marker='o'); plt.xlabel("alpha"); plt.ylabel("accuracy"); plt.title("Accuracy vs alpha"); plt.grid(True); plt.show()

if __name__ == "__main__":
    main()
