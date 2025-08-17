# ======= Context-Prior Logit Fusion on Oxford-IIIT Pets (MVP) =======
# 베이스: CLIP zero-shot
# 보정: BERT 기반 라벨-문장 컨텍스트(z1 pooling) -> 선형사상 M -> 로짓 가산
# 모델 3개 사용버전(image:res, text: CLIP_image, BERT_text <- 두 모델을 이용해서 pior만듬)

import os, json, math, torch, numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import Ridge
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import open_clip
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH = 128
BERT_NAME = "bert-base-uncased"   # z 컨텍스트 추출용 (경량)
CLIP_BACKBONE = "ViT-B-32"        # 가벼운 CLIP
CLIP_PRETRAIN = "openai"
ALPHAS = [0.0, 0.1, 0.2, 0.5, 1.0]  # 로짓 보정 강도 스윕
RIDGE_LAMBDA = 1e-2               # BERT->CLIP 정렬용 Ridge 가중치

# ---------- 0) 데이터 ----------
def get_loader():
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],
                             std=[0.26862954,0.26130258,0.27577711]),
    ])
    test = OxfordIIITPet("./data", split="test", download=True, transform=tfm)
    from torch.utils.data import DataLoader
    dl = DataLoader(test, batch_size=BATCH, shuffle=False, num_workers=0)
    classes = [c.replace("_"," ") for c in test.classes]
    return test, dl, classes

# ---------- 1) CLIP 로드 ----------
def load_clip():
    model, _, _ = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_PRETRAIN, device=DEVICE)
    model.eval()
    tok = open_clip.get_tokenizer(CLIP_BACKBONE)
    return model, tok

# ---------- 2) 이미지 임베딩 ----------
@torch.no_grad()
def img_embeds(clip_model, loader):
    xs, ys = [], []
    for x, y in tqdm(loader, desc="Image embeds"):
        x = x.to(DEVICE)
        with torch.cuda.amp.autocast():
            e = clip_model.encode_image(x)
        xs.append(F.normalize(e, dim=-1).float().cpu())
        ys.append(y)
    return torch.cat(xs,0), torch.cat(ys,0)

# ---------- 3) 라벨 문장 사전(기본 템플릿) ----------
# *여기에 네가 직접 쓴 문장들로 덮어쓰면 성능↑*
def default_prompts_for(classes):
    # 시각적 단서 템플릿 8개 (라벨명을 끼워 넣음)
    tmpl = [
        "{c} has distinctive coat patterns.",
        "{c} shows a characteristic ear shape.",
        "{c} typically has specific eye shape and color.",
        "{c} has a typical body size and proportions.",
        "{c} has a recognizable muzzle and face structure.",
        "{c} has a notable tail shape and carriage.",
        "{c} coat length and texture help identify it.",
        "The facial mask pattern of {c} is a key trait."
    ]
    d = {}
    for c in classes:
        d[c] = [t.format(c=c) for t in tmpl]
    # 예시: 네가 준 Siamese 문장으로 덮어쓰기
    if "American Pit Bull Terrier" in d:
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
    elif "Birman" in d:
        d["Birman"] = [
            "Birman has a silky medium-long coat with a pale cream body.",
            "Birman shows darker points on the face ears legs and tail.",
            "Birman has bright blue eyes that are round and expressive.",
            "Birman has white gloves on all four paws.",
            "Birman has medium-sized ears with slightly rounded tips.",
            "Birman has a sweet rounded facial appearance.",
            "Birman has a bushy tail proportional to the body length.",
            "Birman has a soft coat that lies close without matting easily."
        ]
    elif "Chihuahua" in d:
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
    elif "Staffordshire Bull Terrier" in d:
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
    elif "Wheaten Terrier" in d:
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
    elif "Saint Bernard" in d:
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
    elif "Ragdoll" in d:
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
    elif "Pomeranian" in d:
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
    elif "Beagle" in d:
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
    elif "English Setter" in d:
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


    return d

# ---------- 4) BERT에서 라벨 토큰 컨텍스트(z) 뽑기 ----------
@torch.no_grad()
def build_label_contexts(prompts_dict, layer_idx=8):
    tok = AutoTokenizer.from_pretrained(BERT_NAME)
    mdl = AutoModel.from_pretrained(BERT_NAME).to(DEVICE).eval()
    names, vecs = [], []
    for cls, sents in prompts_dict.items():
        # 문장별로 라벨 토큰 위치 hidden 추출 -> 평균
        sent_vs = []
        for s in sents:
            enc = tok(s, return_tensors="pt").to(DEVICE)
            out = mdl(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx][0]   # (T, d_bert)
            # 라벨 토큰 subword 시퀀스 찾기
            cls_ids = tok(cls, add_special_tokens=False)["input_ids"]
            ids = enc["input_ids"][0].tolist()
            pos = -1
            for i in range(len(ids)-len(cls_ids)+1):
                if ids[i:i+len(cls_ids)] == cls_ids:
                    pos = i; break
            if pos == -1:
                v = hs.mean(0)                # fallback: 문장 평균
            else:
                v = hs[pos:pos+len(cls_ids)].mean(0)
            sent_vs.append(v)
        v_cls = torch.stack(sent_vs, 0).mean(0)
        v_cls = F.normalize(v_cls, dim=-1)
        names.append(cls); vecs.append(v_cls.cpu())
    C = torch.stack(vecs, 0)      # (K, d_bert)
    return names, C

# ---------- 5) CLIP 텍스트 임베딩 ----------
@torch.no_grad()
def clip_text_embeds(clip_model, clip_tok, classes):
    texts = [f"a photo of a {c}" for c in classes]
    toks = clip_tok(texts).to(DEVICE)
    with torch.cuda.amp.autocast():
        t = clip_model.encode_text(toks)
    return F.normalize(t, dim=-1).float().cpu()  # (K, d_clip)

# ---------- 6) BERT 컨텍스트 -> CLIP 텍스트 공간 정렬 (클래스 단위 Ridge) ----------
def fit_align_matrix(C_bert, T_clip, ridge_lambda=1e-2):
    # Solve: minimize || C M - T ||_F^2 + λ||M||_F^2
    # Closed-form via scikit Ridge by row, or one-shot:
    ridge = Ridge(alpha=ridge_lambda, fit_intercept=False)
    # Fit separate regressor for each dim of T
    M = []
    X = C_bert.numpy()
    Y = T_clip.numpy()
    for j in range(Y.shape[1]):
        ridge.fit(X, Y[:, j])
        M.append(ridge.coef_[None, :])  # (1, d_bert)
    M = np.vstack(M)  # (d_clip, d_bert)
    M = torch.tensor(M, dtype=torch.float32)  # maps: (d_clip, d_bert); use (C @ M^T) to get clip-dim
    return M

# ---------- 7) 평가 ----------
def main():
    # Data & CLIP
    testset, loader, classes = get_loader()
    clip_model, clip_tok = load_clip()

    # Image embeddings
    X_img, y = img_embeds(clip_model, loader)

    # Text (CLIP) for baseline
    T_clip = clip_text_embeds(clip_model, clip_tok, classes)     # (K, d_clip)
    logits_base = X_img @ T_clip.T                               # (N, K)

    # Our label contexts (BERT z pooling)
    prompts = default_prompts_for(classes)
    names_ctx, C_bert = build_label_contexts(prompts, layer_idx=8)  # (K, d_bert)
    assert names_ctx == classes, "클래스 순서가 다르면 정렬 맞춰주세요."

    # Align BERT->CLIP text space
    M = fit_align_matrix(C_bert, T_clip, ridge_lambda=RIDGE_LAMBDA)  # (d_clip, d_bert)
    C_mapped = F.normalize(C_bert @ M.T, dim=-1)                     # (K, d_clip)

    # Prior logits from contexts
    logits_prior = X_img @ C_mapped.T                                # (N, K)

    # Evaluate across alphas
    print("\n=== Zero-shot CLIP vs Ours(Logit Fusion) on Pets (test split) ===")
    for a in ALPHAS:
        logits = logits_base if a == 0.0 else (logits_base + a * logits_prior)
        pred = logits.argmax(dim=1).numpy()
        acc = accuracy_score(y.numpy(), pred)
        print(f"alpha={a:.2f}  Top-1={acc*100:.2f}%")

    # Bonus: worst 15 classes by (precision+recall+f1)/3 at best alpha
    best_a = max(ALPHAS, key=lambda aa: accuracy_score(y.numpy(), (logits_base + aa*logits_prior if aa>0 else logits_base).argmax(1).numpy()))
    logits = logits_base if best_a==0 else (logits_base + best_a*logits_prior)
    pred = logits.argmax(1).numpy()
    print(f"\nBest alpha: {best_a:.2f}")
    rep = classification_report(y.numpy(), pred, target_names=classes, output_dict=True, zero_division=0)

    rows=[]
    for cls in classes:
        p, r, f1 = rep[cls]["precision"], rep[cls]["recall"], rep[cls]["f1-score"]
        avg = (p+r+f1)/3
        rows.append((cls, p, r, f1, avg))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["class","precision","recall","f1","avg"])
    df = df.sort_values("avg")
    print("\n=== Worst 15 classes by (P+R+F1)/3 at best alpha ===")
    for _, row in df.head(15).iterrows():
        print(f"{row['class']:<25} avg={row['avg']*100:5.1f}%  "
              f"(P={row['precision']*100:4.1f}  R={row['recall']*100:4.1f}  F1={row['f1']*100:4.1f})")

if __name__ == "__main__":
    main()
