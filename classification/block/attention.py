import torch, torch.nn as nn

class AttentionFusionModel(nn.Module):
    def __init__(self, image_feat_dim, text_feat_dim, num_classes, num_heads=8, num_layers=4):
        super().__init__()
        self.num_classes = num_classes

        # 이미지 특징과 텍스트 특징의 차원을 맞추는 MLP (선택 사항이지만 유용)
        # BERT와 ResNet의 특징 차원이 다를 경우, 하나의 공통 차원으로 맞춰줍니다.
        self.image_proj = nn.Linear(image_feat_dim, image_feat_dim)
        self.text_proj = nn.Linear(text_feat_dim, image_feat_dim) # 텍스트 차원을 이미지 차원으로 투영
        self.common_dim = image_feat_dim # 공통 특징 차원 (여기서는 image_feat_dim 사용)

        # 교차 어텐션 레이어 (Transformer Encoder Layer 사용)
        # 이미지 특징을 쿼리(Query), 텍스트 특징을 키/밸류(Key/Value)로 사용하는 교차 어텐션
        # 혹은, 이미지와 텍스트를 Concat하여 Multi-Head Self-Attention을 적용할 수도 있습니다.
        
        # 여기서는 이미지 특징을 '쿼리'로, 텍스트 특징을 '키/밸류'로 보고, 
        # 이미지 특징을 중심으로 텍스트 정보를 통합하는 방식으로 설계합니다.
        
        # TransformerEncoderLayer는 self-attention이 기본이지만,
        # 여기에 교차 어텐션 로직을 직접 구현하거나,
        # nn.TransformerDecoderLayer를 사용하면 cross-attention을 쉽게 구현할 수 있습니다.
        # 편의상 여기서는 이미지와 텍스트 특징을 병합(concatenate)한 후
        # Multi-Head Self-Attention (MHSA)을 적용하는 방식을 채택합니다.
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.common_dim, # 이미지와 텍스트 특징을 합친 후의 공통 차원
            nhead=num_heads,
            dim_feedforward=self.common_dim * 4,
            batch_first=True # batch dimension first
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 최종 분류를 위한 헤드
        self.classifier_head = nn.Sequential(
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.BatchNorm1d(self.common_dim // 2), # 
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(self.common_dim // 2, num_classes)
        )

    def forward(self, image_features, text_features_Cmap):
        # 1. 특징 차원 맞추기 (옵션)
        # text_features_Cmap은 이미 C_map을 통해 이미지 공간으로 매핑된 벡터이므로,
        # 여기서는 image_features와 차원이 동일하다고 가정하고 text_proj는 생략합니다.
        # 만약 BERT 원본 특징 (C_bert)을 직접 사용한다면 text_proj가 필요합니다.
        
        # (N, d_img)
        projected_image_features = self.image_proj(image_features)
        
        # (K, d_img) -> (1, K, d_img)로 확장 후 (N, K, d_img)로 복제
        # 각 이미지 샘플이 모든 클래스 텍스트 특징을 참조할 수 있도록 준비
        # N: 배치 사이즈, K: 클래스 개수, d_img: 특징 차원
        batch_size = image_features.shape[0]
        expanded_text_features = text_features_Cmap.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 이미지 특징을 쿼리, 텍스트 특징을 키/밸류로 교차 어텐션
        # 여기서는 간단히 이미지 특징과 텍스트 특징을 concatenate 한 후 MHSA 적용.
        # 이 경우, 이미지 특징은 "이미지 토큰"으로, 텍스트 특징은 "텍스트 토큰"으로 간주됩니다.
        # 즉, (이미지 토큰 1개 + 텍스트 토큰 K개)의 시퀀스 입력
        
        # 이미지 특징 (N, d_img) -> (N, 1, d_img)로 unsqueeze
        # 텍스트 특징 (N, K, d_img)
        
        # 입력 시퀀스 구성: [CLS_token (or global_image_token), Text_Tokens...]
        # 여기서는 이미지 특징 자체를 'CLS 토큰'처럼 활용하여
        # 텍스트 특징들과 함께 시퀀스로 만듭니다.
        
        # (N, 1, common_dim) + (N, K, common_dim) -> (N, 1+K, common_dim)
        fusion_input = torch.cat([projected_image_features.unsqueeze(1), expanded_text_features], dim=1)
        
        # Transformer Fusion
        fused_features = self.transformer_fusion(fusion_input) # (N, 1+K, common_dim)
        
        # 3. 최종 분류를 위해 이미지에 해당하는 융합된 특징만 추출 (첫 번째 토큰)
        # 이미지 특징이었던 첫 번째 토큰이 텍스트 정보를 통합한 융합 특징이 됩니다.
        fused_image_representation = fused_features[:, 0, :] # (N, common_dim)
        
        # 4. 분류 헤드에 통과시켜 로짓 출력
        final_logits = self.classifier_head(fused_image_representation)
        
        return final_logits