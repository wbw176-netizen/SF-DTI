import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
import math

def binary_cross_entropy(pred_output, labels):
    # 使用BCEWithLogitsLoss代替BCELoss，使其支持AMP
    loss_fct = torch.nn.BCEWithLogitsLoss()
    n = torch.sigmoid(torch.squeeze(pred_output, 1))
    loss = loss_fct(torch.squeeze(pred_output, 1), labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    # 确保线性输出至少有2个类别
    if linear_output.size(1) < 2:
        # 如果只有一个输出，则扩展为两个类别 [1-p, p]
        p = torch.sigmoid(linear_output)
        linear_output = torch.cat([1-p, p], dim=1)
    
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]  # 获取正类的概率
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    """计算Logits的熵"""
    probs = F.softmax(linear_output, dim=1)
    log_probs = F.log_softmax(linear_output, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()



class SCMAF(nn.Module):
    """
    Subspace-Constrained Mutual Alignment Fusion for drug features.

    It fuses pooled graph features with pretrained chemical features by
    projecting both modalities into a shared space, applying bidirectional
    alignment, projecting through a shared subspace basis, and preserving
    modality-specific residual information.
    """

    def __init__(self, model_feat_dim, precomputed_feat_dim, fusion_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.scales = [1]

        self.structure_path = nn.Sequential(
            nn.Linear(model_feat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.property_path = nn.Sequential(
            nn.Linear(precomputed_feat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        self.ot_alignment = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )

        self.subspace_proj_gcn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.subspace_proj_chem = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )
        self.subspace_basis = nn.Parameter(torch.randn(fusion_dim, fusion_dim) * 0.02)

        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
        )
        self.layer_norm = nn.LayerNorm(fusion_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        with torch.no_grad():
            if hasattr(torch, "linalg"):
                q, _ = torch.linalg.qr(self.subspace_basis)
            else:
                q, _ = torch.qr(self.subspace_basis)
            self.subspace_basis.data = q

    def _optimal_transport_alignment(self, feat_a, feat_b):
        combined = torch.cat([feat_a, feat_b], dim=-1)
        return self.ot_alignment(combined)

    def _subspace_projection(self, feat, proj_type="gcn"):
        basis = F.normalize(self.subspace_basis, p=2, dim=1)
        projection = torch.matmul(basis, basis.T)
        feat_proj = self.subspace_proj_gcn(feat) if proj_type == "gcn" else self.subspace_proj_chem(feat)
        return torch.matmul(feat_proj, projection.T)

    def forward(self, model_features, precomputed_features):
        f_gcn = self.structure_path(model_features)
        f_chem = self.property_path(precomputed_features)

        f_gcn_ot = self._optimal_transport_alignment(f_gcn, f_chem)
        f_chem_ot = self._optimal_transport_alignment(f_chem, f_gcn)

        f_gcn_proj = self._subspace_projection(f_gcn_ot, "gcn")
        f_chem_proj = self._subspace_projection(f_chem_ot, "chem")

        f_weighted = 0.5 * f_gcn_proj + 0.5 * f_chem_proj
        f_fused = self.final_fusion(torch.cat([f_gcn_proj, f_chem_proj], dim=-1))

        return self.layer_norm(f_fused + f_weighted + f_gcn + f_chem)


class MPMI(nn.Module):
    
    def __init__(self, prott5_dim=1024, esm2_dim=1280, fusion_dim=1024, 
                 dropout=0.1, num_heads=4, ablation_config=None):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # 消融实验配置
        default_ablation = {
            'use_learnable_scale': True,    # 可学习缩放参数
            'use_gating': True,             # 门控机制
            'use_residual': True,           # 残差压缩通路
        }
        if ablation_config:
            default_ablation.update(ablation_config)
        self.ablation = default_ablation
        
        # 模态特定轻量适配器 + 残差压缩通路（降低信息损失）
        self.prott5_adapter = nn.Sequential(
            nn.LayerNorm(prott5_dim),
            nn.Linear(prott5_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        self.esm2_adapter = nn.Sequential(
            nn.LayerNorm(esm2_dim),
            nn.Linear(esm2_dim, fusion_dim * 2),
            nn.GELU(),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # 残差压缩通路（根据消融配置）
        if self.ablation['use_residual']:
            self.prott5_residual_compress = nn.Linear(prott5_dim, fusion_dim, bias=False)
            self.esm2_residual_compress = nn.Linear(esm2_dim, fusion_dim, bias=False)
        
        # 门控权重（根据消融配置）
        if self.ablation['use_gating']:
            self.prott5_gate = nn.Parameter(torch.tensor(0.7))
            self.esm2_gate = nn.Parameter(torch.tensor(0.7))
        
        # 投影后规范化
        self.prott5_proj_norm = nn.LayerNorm(fusion_dim)
        self.esm2_proj_norm = nn.LayerNorm(fusion_dim)
        
        # 可学习缩放（根据消融配置）
        if self.ablation['use_learnable_scale']:
            self.prott5_scale = nn.Parameter(torch.tensor(1.0))
            self.esm2_scale = nn.Parameter(torch.tensor(1.0))
        
        self.shared_interaction = SharedMutualInteraction(fusion_dim, num_heads, dropout)
        
        # 最终融合层（降低Dropout率，避免过度正则化）
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)  # 降低Dropout率：0.1 -> 0.05
        )
        
    def forward(self, prott5_feat, esm2_feat):
        """
        Args:
            prott5_feat: [batch_size, prott5_dim]
            esm2_feat: [batch_size, esm2_dim]
        Returns:
            fused_feat: [batch_size, fusion_dim]
        """
        if prott5_feat.dim() != 2 or esm2_feat.dim() != 2:
            raise ValueError("MPMI expects pooled 2D protein features with shape [B, D].")
        return self._fuse_pooled_features(prott5_feat, esm2_feat)
    
    def _fuse_pooled_features(self, prott5_feat, esm2_feat):
        """融合池化特征（2D输入）"""
        # 模态特定适配
        p_main = self.prott5_adapter(prott5_feat)
        e_main = self.esm2_adapter(esm2_feat)
        
        # 根据消融配置处理特征
        if self.ablation['use_residual'] and self.ablation['use_gating']:
            # 完整版本：残差压缩 + 门控
            p_res = self.prott5_residual_compress(prott5_feat)
            e_res = self.esm2_residual_compress(esm2_feat)
            prott5_proj = self.prott5_proj_norm(self.prott5_gate * p_main + (1 - self.prott5_gate) * p_res)
            esm2_proj = self.esm2_proj_norm(self.esm2_gate * e_main + (1 - self.esm2_gate) * e_res)
        elif self.ablation['use_residual'] and not self.ablation['use_gating']:
            # 移除门控：只使用残差压缩
            p_res = self.prott5_residual_compress(prott5_feat)
            e_res = self.esm2_residual_compress(esm2_feat)
            prott5_proj = self.prott5_proj_norm(p_main + p_res)
            esm2_proj = self.esm2_proj_norm(e_main + e_res)
        elif not self.ablation['use_residual'] and self.ablation['use_gating']:
            # 移除残差：只使用门控（但门控失效，因为只有一个分支）
            prott5_proj = self.prott5_proj_norm(p_main)
            esm2_proj = self.esm2_proj_norm(e_main)
        else:
            # 最简版本：只使用主适配器
            prott5_proj = self.prott5_proj_norm(p_main)
            esm2_proj = self.esm2_proj_norm(e_main)
        
        # 可学习缩放（根据消融配置）
        if self.ablation['use_learnable_scale']:
            prott5_proj = self.prott5_scale * prott5_proj
            esm2_proj = self.esm2_scale * esm2_proj
        
        fused_feat = self.shared_interaction(prott5_proj, esm2_proj)
        return self.final_fusion(fused_feat)
    
class SharedMutualInteraction(nn.Module):
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"

        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.dropout_p = dropout
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.residual_weight = 0.5

    def forward(self, x1, x2):
        output_a = self._project_and_drop(x2)
        output_b = self._project_and_drop(x1)
        fused_feat = output_a + output_b
        fused_feat = fused_feat + self.residual_weight * (x1 + x2)
        return self.layer_norm(fused_feat)

    def _project_and_drop(self, x):
        batch_size = x.size(0)
        v = self.v_proj(x).view(batch_size, self.num_heads, self.head_dim)

        if self.training and self.dropout_p > 0:
            keep_prob = 1.0 - self.dropout_p
            mask = torch.empty(
                batch_size, self.num_heads, 1,
                device=v.device, dtype=v.dtype
            ).bernoulli_(keep_prob).div_(keep_prob)
            v = v * mask

        return v.reshape(batch_size, self.feat_dim)


class EfficientMCA(nn.Module):
    
    def __init__(self, feat_dim, num_heads, dropout=0.1, qk_ratio=0.5):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        # 值向量高维保真：仅 Q/K 降维，V 保持高维
        self.head_dim = feat_dim // num_heads  # V 的每头维度
        qk_dim = max(1, int(feat_dim * qk_ratio))
        # 确保能被 num_heads 整除
        qk_dim = (qk_dim // num_heads) * num_heads
        if qk_dim == 0:
            qk_dim = num_heads  # 至少每头1维
        self.qk_dim = qk_dim
        self.qk_head_dim = self.qk_dim // num_heads
        
        assert feat_dim % num_heads == 0, "feat_dim must be divisible by num_heads"
        assert self.qk_dim % num_heads == 0, "qk_dim must be divisible by num_heads"
        
        # 简化的投影层（无偏置）
        self.q_proj = nn.Linear(feat_dim, self.qk_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, self.qk_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feat_dim)
        
        # 固定的残差权重
        self.residual_weight = 0.5
        
    def forward(self, x1, x2):
        """
        Args:
            x1: [batch_size, seq_len, feat_dim]
            x2: [batch_size, seq_len, feat_dim]
        Returns:
            fused_feat: [batch_size, seq_len, feat_dim]
        """
        batch_size, seq_len, _ = x1.shape
        
        # 简化的双向注意力
        # x1 -> x2
        output_A = self._efficient_attention(x1, x2, x2)
        
        # x2 -> x1  
        output_B = self._efficient_attention(x2, x1, x1)
        
        # 双向融合 + 残差连接
        fused_feat = output_A + output_B
        fused_feat = fused_feat + self.residual_weight * (x1 + x2)
        
        # 层归一化
        fused_feat = self.layer_norm(fused_feat)
        
        return fused_feat
    
    def _efficient_attention(self, query, key, value):
        """高效注意力计算"""
        batch_size, seq_len, _ = query.shape
        
        # 投影到Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.feat_dim
        )
        
        return attn_output


class CMSPF(nn.Module):
    """Cross-Modal Signal Propagation Fusion (CMSPF) - Enhanced CMRFF
    基于SiFu机制的跨模态信号传播融合模块：
      - 静态跨模态语义映射：将药物-蛋白质交互模式映射为语义节点
      - 动态信号传播：通过信号能量在跨模态节点间传播
      - 双向条件调制：对方模态生成信号调制参数
      - 路由式交互：基于信号能量的特征路由与融合
      - 多尺度空洞卷积空间建模，轻量SE通道注意力
      - 层级差异化级联结构与最终多尺度融合，接口与输出形状与原模块一致
    forward(x_drug, x_protein, x_shared=None) -> (x_drug_out, x_protein_out, x_shared_out)
    """
    def __init__(self, nf=128, num_heads=8, dropout=0.1, num_cascades=4):
        super(CMSPF, self).__init__()
        self.nf = nf
        self.num_cascades = num_cascades
        self.dropout = nn.Dropout(dropout)

        # SiFu机制核心组件
        # 1. 静态跨模态语义映射
        self.semantic_nodes = nn.Parameter(torch.randn(nf, nf))  # 跨模态语义节点
        self.node_bias = nn.Parameter(torch.zeros(nf, 1, 1))     # 节点偏置
        
        # 2. 信号传播权重矩阵（边权重）
        self.signal_weights = nn.Parameter(torch.randn(nf, nf))  # (nf, nf) 用于信号传播
        self.signal_bias = nn.Parameter(torch.zeros(nf, 1, 1))       # 信号偏置
        
        # 3. 跨模态交互边权重
        self.cross_modal_weights = nn.Parameter(torch.randn(2*nf, nf))  # 跨模态边权重 (2*nf, nf)
        self.cross_modal_bias = nn.Parameter(torch.zeros(nf, 1, 1))   # 跨模态偏置
        
        # 4. 信号能量激活函数
        self.energy_activation = nn.GELU()
        
        # 层级差异化特征变换
        self.drug_transform = nn.ModuleList()
        self.protein_transform = nn.ModuleList()
        for i in range(num_cascades):
            if i == 0:
                # 第一层：基础特征提取，使用轻量级结构
                self.drug_transform.append(self._make_lightweight_block(nf))
                self.protein_transform.append(self._make_lightweight_block(nf))
            elif i < num_cascades - 1:
                # 中间层：跨模态交互，使用标准结构
                self.drug_transform.append(self._make_res_block(nf))
                self.protein_transform.append(self._make_res_block(nf))
            else:
                # 最后一层：精细融合，使用增强结构
                self.drug_transform.append(self._make_enhanced_block(nf))
                self.protein_transform.append(self._make_enhanced_block(nf))

        # 层级注意力机制
        self.layer_attention = nn.ModuleList()
        for i in range(num_cascades):
            self.layer_attention.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(nf, nf // 4, 1),
                nn.GELU(),
                nn.Conv2d(nf // 4, 1, 1),
                nn.Sigmoid()
            ))
        
        # 信号传播缓存机制
        self.signal_cache = {}
        self.use_signal_cache = True
        
        # 路由门控与条件调制
        self.router = nn.ModuleList()
        for i in range(num_cascades):
            self.router.append(nn.ModuleDict({
                'gap': nn.AdaptiveAvgPool2d(1),
                # 条件调制：从对方模态生成 scale 与 shift
                'cond_d_from_p': nn.Sequential(
                    nn.Conv2d(nf, nf, 1), nn.GELU(), nn.Conv2d(nf, nf * 2, 1)
                ),
                'cond_p_from_d': nn.Sequential(
                    nn.Conv2d(nf, nf, 1), nn.GELU(), nn.Conv2d(nf, nf * 2, 1)
                ),
                # 路由卷积（深度可分离）
                'route_d': nn.Sequential(
                    nn.Conv2d(nf, nf, 3, padding=1, groups=nf),
                    nn.Conv2d(nf, nf, 1), nn.GELU()
                ),
                'route_p': nn.Sequential(
                    nn.Conv2d(nf, nf, 3, padding=1, groups=nf),
                    nn.Conv2d(nf, nf, 1), nn.GELU()
                ),
                # 多尺度空洞深度卷积
                'spatial_mixer_d': nn.ModuleList([
                    nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1)),
                    nn.Sequential(nn.Conv2d(nf, nf, 3, padding=2, dilation=2, groups=nf), nn.Conv2d(nf, nf, 1)),
                    nn.Sequential(nn.Conv2d(nf, nf, 3, padding=3, dilation=3, groups=nf), nn.Conv2d(nf, nf, 1))
                ]),
                'spatial_mixer_p': nn.ModuleList([
                    nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1)),
                    nn.Sequential(nn.Conv2d(nf, nf, 3, padding=2, dilation=2, groups=nf), nn.Conv2d(nf, nf, 1)),
                    nn.Sequential(nn.Conv2d(nf, nf, 3, padding=3, dilation=3, groups=nf), nn.Conv2d(nf, nf, 1))
                ]),
                # 融合与FFN
                'fusion_d': nn.Sequential(
                    nn.Conv2d(nf * 3, nf, 1), nn.GELU(),
                    nn.Conv2d(nf, nf, 3, padding=1), nn.GELU(),
                    nn.Conv2d(nf, nf, 1)
                ),
                'fusion_p': nn.Sequential(
                    nn.Conv2d(nf * 3, nf, 1), nn.GELU(),
                    nn.Conv2d(nf, nf, 3, padding=1), nn.GELU(),
                    nn.Conv2d(nf, nf, 1)
                ),
                'ffn_d': nn.Sequential(
                    nn.Conv2d(nf, nf * 4, 1), nn.GELU(), self.dropout, nn.Conv2d(nf * 4, nf, 1)
                ),
                'ffn_p': nn.Sequential(
                    nn.Conv2d(nf, nf * 4, 1), nn.GELU(), self.dropout, nn.Conv2d(nf * 4, nf, 1)
                ),
                # 轻量通道注意力
                'se_d': nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Conv2d(nf, nf // 8, 1), nn.GELU(), nn.Conv2d(nf // 8, nf, 1), nn.Sigmoid()
                ),
                'se_p': nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Conv2d(nf, nf // 8, 1), nn.GELU(), nn.Conv2d(nf // 8, nf, 1), nn.Sigmoid()
                ),
                # 共享特征更新
                'shared_update': nn.Sequential(
                    nn.Conv2d(nf * 3, nf, 1), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1)
                ),
                'bn_shared': nn.BatchNorm2d(nf)
            }))

        # 多尺度融合与最终聚合
        self.ms_fuse = nn.ModuleDict({
            's1': nn.Conv2d(nf, nf, 1),
            's2': nn.Conv2d(nf, nf, 1),
            's3': nn.Conv2d(nf, nf, 1),
            'combine': nn.Conv2d(nf * 3, nf, 1)
        })
        self.final_fusion = nn.Sequential(
            nn.Conv2d(nf * 3, nf, 1), nn.BatchNorm2d(nf), nn.GELU(),
            nn.Conv2d(nf, nf, 3, padding=1), nn.GELU(),
            nn.Conv2d(nf, nf, 1)
        )

        self._init_weights()

    def _make_res_block(self, nf):
        return nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.GELU(),
            nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf)
        )
    
    def _make_lightweight_block(self, nf):
        """轻量级块：用于第一层基础特征提取"""
        return nn.Sequential(
            nn.Conv2d(nf, nf, 1), nn.BatchNorm2d(nf), nn.GELU(),
            nn.Conv2d(nf, nf, 3, padding=1, groups=nf),  # 深度可分离卷积
            nn.Conv2d(nf, nf, 1), nn.BatchNorm2d(nf)
        )
    
    def _make_enhanced_block(self, nf):
        """增强块：用于最后一层精细融合"""
        return nn.Sequential(
            nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.GELU(),
            nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.GELU(),
            nn.Conv2d(nf, nf, 1), nn.BatchNorm2d(nf)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # SiFu参数初始化
        nn.init.xavier_uniform_(self.semantic_nodes)
        nn.init.xavier_uniform_(self.signal_weights)
        nn.init.xavier_uniform_(self.cross_modal_weights)

    def _signal_propagation(self, x_drug, x_protein, x_shared, layer_idx=0):
        """
        优化的SiFu信号传播机制，支持增量更新
        Args:
            x_drug: (B, C, H, W) 药物特征
            x_protein: (B, C, H, W) 蛋白质特征  
            x_shared: (B, C, H, W) 共享特征
            layer_idx: 当前层索引
        Returns:
            signal_energy_drug: 药物信号能量
            signal_energy_protein: 蛋白质信号能量
            cross_modal_signal: 跨模态信号
        """
        # 使用缓存机制减少重复计算
        if self.use_signal_cache and layer_idx > 0:
            cache_key = f"layer_{layer_idx}"
            if cache_key in self.signal_cache:
                return self.signal_cache[cache_key]
        B, C, H, W = x_drug.shape
        
        # 1. 静态语义节点激活
        # 将特征映射到语义空间
        drug_semantic = torch.matmul(x_drug.view(B, C, -1).transpose(1, 2), self.semantic_nodes)  # (B, H*W, nf)
        protein_semantic = torch.matmul(x_protein.view(B, C, -1).transpose(1, 2), self.semantic_nodes)  # (B, H*W, nf)
        
        # 添加节点偏置 (调整维度匹配)
        drug_semantic = drug_semantic + self.node_bias.squeeze(-1).transpose(0, 1)  # (B, H*W, nf)
        protein_semantic = protein_semantic + self.node_bias.squeeze(-1).transpose(0, 1)  # (B, H*W, nf)
        
        # 2. 信号能量计算（L2范数）
        drug_energy = torch.norm(drug_semantic, dim=-1, keepdim=True)  # (B, H*W, 1)
        protein_energy = torch.norm(protein_semantic, dim=-1, keepdim=True)  # (B, H*W, 1)
        
        # 转换能量为卷积格式 (B, nf, H, W)
        drug_energy = drug_energy.transpose(1, 2).unsqueeze(-1)  # (B, 1, H*W, 1)
        drug_energy = drug_energy.view(B, 1, H, W)  # (B, 1, H, W)
        drug_energy = drug_energy.expand(B, self.nf, H, W)  # (B, nf, H, W)
        
        protein_energy = protein_energy.transpose(1, 2).unsqueeze(-1)  # (B, 1, H*W, 1)
        protein_energy = protein_energy.view(B, 1, H, W)  # (B, 1, H, W)
        protein_energy = protein_energy.expand(B, self.nf, H, W)  # (B, nf, H, W)
        
        # 3. 信号传播
        # 药物信号传播
        drug_signal = self.energy_activation(
            torch.matmul(drug_semantic, self.signal_weights.transpose(0, 1)) + self.signal_bias.squeeze(-1).transpose(0, 1)
        )  # (B, H*W, nf)
        
        # 蛋白质信号传播  
        protein_signal = self.energy_activation(
            torch.matmul(protein_semantic, self.signal_weights.transpose(0, 1)) + self.signal_bias.squeeze(-1).transpose(0, 1)
        )  # (B, H*W, nf)
        
        # 转换信号为卷积格式 (B, nf, H, W)
        drug_signal = drug_signal.transpose(1, 2).unsqueeze(-1)  # (B, nf, H*W, 1)
        drug_signal = drug_signal.view(B, self.nf, H, W)  # (B, nf, H, W)
        
        protein_signal = protein_signal.transpose(1, 2).unsqueeze(-1)  # (B, nf, H*W, 1)
        protein_signal = protein_signal.view(B, self.nf, H, W)  # (B, nf, H, W)
        
        # 4. 跨模态信号计算
        cross_modal_signal = self.energy_activation(
            torch.matmul(
                torch.cat([drug_semantic, protein_semantic], dim=-1), 
                self.cross_modal_weights
            ) + self.cross_modal_bias.squeeze(-1).transpose(0, 1)
        )  # (B, H*W, nf)
        
        # 转换跨模态信号为卷积格式
        cross_modal_signal = cross_modal_signal.transpose(1, 2).unsqueeze(-1)  # (B, nf, H*W, 1)
        cross_modal_signal = cross_modal_signal.view(B, self.nf, H, W)  # (B, nf, H, W)
        
        # 缓存结果
        if self.use_signal_cache:
            cache_key = f"layer_{layer_idx}"
            self.signal_cache[cache_key] = (drug_signal, protein_signal, cross_modal_signal)
        
        return drug_signal, protein_signal, cross_modal_signal

    def _signal_routing(self, x_drug, x_protein, drug_signal, protein_signal, cross_modal_signal):
        """
        基于信号能量的特征路由
        """
        # 信号能量作为路由权重
        drug_energy_map = torch.norm(drug_signal, dim=1, keepdim=True)  # (B, 1, H, W)
        protein_energy_map = torch.norm(protein_signal, dim=1, keepdim=True)  # (B, 1, H, W)
        cross_energy_map = torch.norm(cross_modal_signal, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # 归一化能量权重
        total_energy = drug_energy_map + protein_energy_map + cross_energy_map + 1e-8
        drug_weight = drug_energy_map / total_energy
        protein_weight = protein_energy_map / total_energy
        cross_weight = cross_energy_map / total_energy
        
        # 基于信号能量的特征路由
        # 注意：x_drug和x_protein是C维，而signals是nf维，需要投影
        if x_drug.shape[1] != self.nf:
            # 如果输入特征维度与信号维度不匹配，使用线性投影
            if not hasattr(self, 'drug_proj'):
                self.drug_proj = nn.Linear(x_drug.shape[1], self.nf).to(x_drug.device)
                self.protein_proj = nn.Linear(x_protein.shape[1], self.nf).to(x_protein.device)
            x_drug_proj = self.drug_proj(x_drug.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x_protein_proj = self.protein_proj(x_protein.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x_drug_proj = x_drug
            x_protein_proj = x_protein
            
        routed_drug = x_drug_proj * drug_weight + cross_modal_signal * cross_weight
        routed_protein = x_protein_proj * protein_weight + cross_modal_signal * cross_weight
        
        return routed_drug, routed_protein, cross_modal_signal

    def _ms_features(self, x):
        s1 = self.ms_fuse['s1'](x)
        h, w = x.shape[2], x.shape[3]
        if h >= 2 and w >= 2:
            s2 = F.avg_pool2d(x, 2, 2)
        else:
            s2 = F.avg_pool2d(x, (min(2, h), min(2, w)))
        s2 = self.ms_fuse['s2'](s2)
        s2 = F.interpolate(s2, size=x.shape[2:], mode='bilinear', align_corners=False)
        if h >= 4 and w >= 4:
            s3 = F.avg_pool2d(x, 4, 4)
        else:
            s3 = F.avg_pool2d(x, (min(4, h), min(4, w)))
        s3 = self.ms_fuse['s3'](s3)
        s3 = F.interpolate(s3, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.ms_fuse['combine'](torch.cat([s1, s2, s3], dim=1))

    def forward(self, x_drug, x_protein, x_shared=None):
        if x_shared is None:
            x_shared = (x_drug + x_protein) / 2.0

        x_drug_res, x_protein_res, x_shared_res = x_drug, x_protein, x_shared

        # 清空信号缓存
        if self.use_signal_cache:
            self.signal_cache.clear()

        # SiFu信号传播初始化
        drug_signal, protein_signal, cross_modal_signal = self._signal_propagation(x_drug, x_protein, x_shared, layer_idx=0)

        for i in range(self.num_cascades):
            # 层级差异化处理
            trans_d = self.drug_transform[i](x_drug)
            trans_p = self.protein_transform[i](x_protein)

            rt = self.router[i]
            
            # SiFu信号路由
            routed_d, routed_p, cross_modal_signal = self._signal_routing(
                trans_d, trans_p, drug_signal, protein_signal, cross_modal_signal
            )
            
            # 双向条件调制（对方 -> 本方）
            gd = rt['gap'](routed_d)
            gp = rt['gap'](routed_p)

            cond_d = rt['cond_d_from_p'](gp)
            cond_p = rt['cond_p_from_d'](gd)
            scale_d, shift_d = torch.chunk(cond_d, 2, dim=1)
            scale_p, shift_p = torch.chunk(cond_p, 2, dim=1)
            
            # 信号能量调制
            mod_d = routed_d * torch.sigmoid(scale_d) + shift_d
            mod_p = routed_p * torch.sigmoid(scale_p) + shift_p

            # 路由：使用对方调制后的特征引导
            routed_d = rt['route_d'](mod_p)
            routed_p = rt['route_p'](mod_d)

            # 多尺度空间混合
            ms_d = sum(branch(mod_d) for branch in rt['spatial_mixer_d']) / 3.0
            ms_p = sum(branch(mod_p) for branch in rt['spatial_mixer_p']) / 3.0

            # 轻量通道注意力
            attn_d = rt['se_d'](ms_d)
            attn_p = rt['se_p'](ms_p)
            ms_d = ms_d * attn_d
            ms_p = ms_p * attn_p

            # 融合 + FFN
            fused_d = rt['fusion_d'](torch.cat([trans_d, routed_d, ms_d], dim=1))
            fused_p = rt['fusion_p'](torch.cat([trans_p, routed_p, ms_p], dim=1))
            ffn_d = rt['ffn_d'](fused_d) + fused_d
            ffn_p = rt['ffn_p'](fused_p) + fused_p

            # 层级注意力机制
            layer_attn = self.layer_attention[i](ffn_d + ffn_p)
            ffn_d = ffn_d * layer_attn
            ffn_p = ffn_p * layer_attn

            # 共享更新（集成跨模态信号）
            shared_input = torch.cat([ffn_d, ffn_p, cross_modal_signal], dim=1)
            x_shared = rt['bn_shared'](rt['shared_update'](shared_input)) + x_shared

            # 残差推进
            x_drug = ffn_d + x_drug
            x_protein = ffn_p + x_protein
            
            # 更新信号传播（用于下一层）
            if i < self.num_cascades - 1:
                drug_signal, protein_signal, cross_modal_signal = self._signal_propagation(
                    x_drug, x_protein, x_shared, layer_idx=i+1
                )

        # 多尺度融合与最终聚合
        ms_d = self._ms_features(x_drug)
        ms_p = self._ms_features(x_protein)
        final_shared = self.final_fusion(torch.cat([ms_d, ms_p, x_shared], dim=1))

        return x_drug + x_drug_res, x_protein + x_protein_res, final_shared + x_shared_res

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, 
                      hidden_feats=hidden_feats, 
                      activation=activation,
                      allow_zero_in_degree=True)  # 允许零入度节点
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        
        # 创建节点到图的映射
        batch_num_nodes = batch_graph.batch_num_nodes()
        max_nodes = max(batch_num_nodes).item()
        
        # 创建输出张量并填充
        output = torch.zeros(batch_size, max_nodes, self.output_feats, device=node_feats.device)
        
        # 填充张量
        start_idx = 0
        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            output[i, :num_nodes, :] = node_feats[start_idx:start_idx+num_nodes, :]
            start_idx += num_nodes
            
        return output

def get_freq_indices(method):
    """获取频率索引"""
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class NonlinearFrequencyTransform(nn.Module):
    """非线性频率变换模块 - 优化版本：大幅提升训练速度"""
    
    def __init__(self, in_channels, transform_type='swish', num_bases=4):
        super(NonlinearFrequencyTransform, self).__init__()
        
        self.in_channels = in_channels
        self.transform_type = transform_type
        self.num_bases = num_bases
        
        # 只保留最有效的非线性变换参数
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        
        # 简化基函数数量，从8减少到4
        self.frequency_bases = nn.Parameter(torch.randn(num_bases, in_channels) * 0.1)
        self.basis_weights = nn.Parameter(torch.ones(num_bases))
        
        # 只保留最有效的激活函数
        self.nonlinear_activations = nn.ModuleDict({
            'swish': nn.SiLU(),      # 最常用且高效
            'gelu': nn.GELU(),       # 性能最好
            'relu': nn.ReLU(inplace=True)  # 最简单快速
        })
        
        # 简化通道重建网络
        self.channel_reconstructor = nn.Conv2d(1, in_channels, 1, bias=False)
        
        # 预定义通道重建器字典
        self.channel_reconstructors = {}
    
    def polynomial_transform(self, x, degree=2):  # 从3减少到2
        """简化多项式变换 - 只计算到2次方"""
        return x + (self.alpha ** 2) * (x ** 2)
    
    def adaptive_basis_transform(self, x):
        """优化的自适应基函数变换"""
        bases = self.frequency_bases
        weights = F.softmax(self.basis_weights, dim=0)
        
        # 使用更高效的矩阵运算
        x_reshaped = x.view(x.size(0), x.size(1), -1)
        similarities = torch.einsum('nc,bch->nbh', bases, x_reshaped)
        
        # 向量化加权组合
        weighted_response = torch.einsum('n,nbh->bh', weights, similarities)
        weighted_response_reshaped = weighted_response.view(x.size(0), 1, x.size(2), x.size(3))
        
        # 使用预定义的通道重建器
        channels = x.size(1)
        if channels not in self.channel_reconstructors:
            self.channel_reconstructors[channels] = nn.Conv2d(1, channels, 1, bias=False).to(x.device)
        
        return self.channel_reconstructors[channels](weighted_response_reshaped)
    
    def forward(self, x, transform_method=None):
        """优化的前向传播 - 只保留最有效的方法"""
        if transform_method is None:
            transform_method = self.transform_type
        
        if transform_method == 'polynomial':
            return self.polynomial_transform(x)
        elif transform_method == 'adaptive':
            return self.adaptive_basis_transform(x)
        elif transform_method == 'swish':
            return self.nonlinear_activations['swish'](x)
        elif transform_method == 'gelu':
            return self.nonlinear_activations['gelu'](x)
        elif transform_method == 'relu':
            return self.nonlinear_activations['relu'](x)
        else:
            # 默认使用swish
            return self.nonlinear_activations['swish'](x)

class EnhancedMultiFrequencyChannelAttention(nn.Module):
    """增强型多频率通道注意力模块 - 优化版本：大幅提升训练速度"""
    
    def __init__(self, in_channels, dct_h=7, dct_w=7, frequency_branches=8,  # 从16减少到8
                 frequency_selection='top', reduction=16, use_nonlinear=True):
        super(EnhancedMultiFrequencyChannelAttention, self).__init__()
        
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)
        
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.use_nonlinear = use_nonlinear
        
        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        
        # 非线性频率变换模块 - 简化版本
        if use_nonlinear:
            self.nonlinear_transform = NonlinearFrequencyTransform(
                in_channels, transform_type='swish', num_bases=4  # 从8减少到4
            )
            
            # 简化变换选择器 - 只选择3种最有效的方法
            self.transform_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 3, 1),  # 从5减少到3
                nn.Softmax(dim=1)
            )
        
        # 预计算DCT权重 - 避免每次前向传播重新计算
        self.register_buffer('dct_weights', self._precompute_dct_weights(mapper_x, mapper_y, in_channels))
        
        # 通道注意力层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # 池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 特征融合
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        
        # 频率响应学习
        self.freq_response_learner = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def _precompute_dct_weights(self, mapper_x, mapper_y, in_channels):
        """预计算DCT权重 - 避免运行时计算"""
        dct_weights = []
        for freq_idx in range(self.num_freq):
            dct_weight = torch.zeros(in_channels, in_channels, self.dct_h, self.dct_w)
            
            for t_x in range(self.dct_h):
                for t_y in range(self.dct_w):
                    for ch in range(in_channels):
                        dct_weight[ch, ch, t_x, t_y] = self._build_filter(t_x, mapper_x[freq_idx], self.dct_h) * \
                                                       self._build_filter(t_y, mapper_y[freq_idx], self.dct_w)
            dct_weights.append(dct_weight)
        
        return torch.stack(dct_weights, dim=0)  # [num_freq, in_channels, in_channels, dct_h, dct_w]
    
    def _build_filter(self, pos, freq, size):
        """构建DCT滤波器"""
        result = math.cos(math.pi * freq * (2 * pos + 1) / (2 * size)) / math.sqrt(size)
        if freq == 0:
            result = result * math.sqrt(2)
        return result
    
    def forward(self, x):
        """优化的前向传播 - 大幅减少计算复杂度"""
        b, c, h, w = x.size()
        
        # 多频率特征提取 - 使用预计算的权重
        freq_features = []
        for freq_idx in range(self.num_freq):
            dct_weight = self.dct_weights[freq_idx]
            freq_feat = F.conv2d(x, dct_weight, padding=(self.dct_h//2, self.dct_w//2), stride=1)
            
            # 应用非线性频率变换 - 简化版本
            if self.use_nonlinear:
                # 只选择3种最有效的变换方法
                transform_weights = self.transform_selector(freq_feat)  # [B, 3, 1, 1]
                
                # 只应用3种变换：swish, gelu, adaptive
                transform_methods = ['swish', 'gelu', 'adaptive']
                nonlinear_features = []
                
                for i, method in enumerate(transform_methods):
                    transformed = self.nonlinear_transform(freq_feat, method)
                    weighted = transformed * transform_weights[:, i:i+1, :, :]
                    nonlinear_features.append(weighted)
                
                # 融合非线性特征
                freq_feat = sum(nonlinear_features)
            
            freq_features.append(freq_feat)
        
        # 频率特征融合 - 使用更高效的方式
        freq_fused = torch.stack(freq_features, dim=1).mean(dim=1)
        
        # 频率响应学习
        freq_response = self.freq_response_learner(freq_fused)
        freq_fused = freq_fused * freq_response
        
        # 通道注意力
        avg_out = self.fc(self.avg_pool(freq_fused))
        max_out = self.fc(self.max_pool(freq_fused))
        attention = torch.sigmoid(avg_out + max_out)
        
        # 应用注意力
        out = x * attention
        
        # 特征增强
        enhanced = torch.cat([out, freq_fused], dim=1)
        out = self.fusion_conv(enhanced)
        
        return out

class PMSAFI(nn.Module):
    
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(PMSAFI, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        
        self.mfca = EnhancedMultiFrequencyChannelAttention(
            in_channels=in_ch[0], 
            frequency_branches=8, 
            frequency_selection='top',
            reduction=16,
            use_nonlinear=True  # 启用非线性频率变换
        )
        
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        
        # 特征对齐层 - 修复通道数不匹配问题
        self.feature_align = nn.Sequential(
            nn.Conv2d(in_ch[0], in_ch[1], 1),  # 从embedding_dim到num_filters
            nn.BatchNorm2d(in_ch[1]),
            nn.ReLU()
        )
        
        # 序列建模层 - 修复通道数不匹配问题
        self.sequence_modeling = nn.Sequential(
            nn.Conv1d(in_ch[0], in_ch[1], kernel_size=3, padding=1),  # 从embedding_dim到num_filters
            nn.BatchNorm1d(in_ch[1]),
            nn.ReLU(),
            nn.Conv1d(in_ch[1], in_ch[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(in_ch[1]),
            nn.ReLU()
        )

    def forward(self, v):
        # 输入处理
        v = self.embedding(v.long())  # [B, seq_len, embedding_dim]
        v = v.transpose(2, 1)  # [B, embedding_dim, seq_len]
        
        # MFCA特征增强
        v = v.unsqueeze(-1)  # [B, embedding_dim, seq_len, 1]
        v = self.mfca(v)  # 多频率通道注意力
        
        # 序列建模
        v = v.squeeze(-1)  # [B, embedding_dim, seq_len]
        v = self.sequence_modeling(v)  # 序列卷积建模
        
        # 特征对齐
        v = v.unsqueeze(-1)  # [B, embedding_dim, seq_len, 1]
        v = self.feature_align(v)
        
        return v  # [B, embedding_dim, seq_len, 1]

class MPRC(nn.Module):
    """创新的双路径特征解码器，结合轴向特征提取与残差连接"""

    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout_rate=None):
        super(MPRC, self).__init__()
        # 主要特征处理路径
        self.proj1 = nn.Linear(in_dim, in_dim)  # 第一个轴向投影
        self.bn1 = nn.BatchNorm1d(in_dim)

        # 辅助特征处理路径
        self.proj2 = nn.Linear(in_dim, in_dim)  # 第二个轴向投影
        self.bn2 = nn.BatchNorm1d(in_dim)

        # 特征融合处理
        self.fuse = nn.Linear(in_dim * 3, hidden_dim)  # 融合不同轴向的特征
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)

        # 输出层
        self.hidden_fc = nn.Linear(hidden_dim, out_dim)
        self.bn_hidden = nn.BatchNorm1d(out_dim)
        self.output_fc = nn.Linear(out_dim, binary)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 保存原始输入作为残差连接
        x_identity = x

        # 第一轴向处理
        x_axis1 = self.relu(self.bn1(self.proj1(x)))

        # 第二轴向处理
        x_axis2 = self.relu(self.bn2(self.proj2(x)))

        # 特征级联并融合
        x_concat = torch.cat([x_axis1, x_axis2, x_identity], dim=1)
        x_fused = self.relu(self.bn_fuse(self.fuse(x_concat)))

        # 最终特征处理和输出（只在最终输出前使用Dropout）
        x_hidden = self.dropout(self.relu(self.bn_hidden(self.hidden_fc(x_fused))))
        output = self.output_fc(x_hidden)

        return output


DualPathFeatureDecoder = MPRC

class SF_DTI(nn.Module):
    def __init__(self, device='cuda', use_precomputed_features=None, **config):
        super(SF_DTI, self).__init__()
        self.device = device
        self.use_precomputed_features = True if use_precomputed_features is None else use_precomputed_features

        def cfg_value(cfg, key, default=None):
            try:
                return cfg[key]
            except (KeyError, TypeError):
                return default

        signal_cfg = cfg_value(config, "SIGNAL")
        if signal_cfg is None:
            signal_cfg = cfg_value(config, "CBIE_CROSS_ATTENTION")
        if signal_cfg is None:
            raise KeyError("Missing SIGNAL config section")

        pretrained_cfg = cfg_value(config, "PRETRAINED", {})
        drug_fusion_cfg = cfg_value(config, "DRUG_FUSION", {})
        protein_fusion_cfg = cfg_value(config, "PROTEIN_FUSION", {})

        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        mlp_dropout_rate = config["DECODER"]["DROPOUT_RATE"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config["PROTEIN"]["NUM_HEAD"]

        cmspf_num_head = signal_cfg["NUM_HEAD"]
        cmspf_emb_dim = signal_cfg["EMBEDDING_DIM"]
        cmspf_num_cascades = signal_cfg["NUM_CASCADES"]
        cmspf_dropout_rate = cfg_value(
            signal_cfg,
            "SIGNAL_DROPOUT_RATE",
            cfg_value(signal_cfg, "CBIE_DROPOUT_RATE", 0.1),
        )
        self.feature_dim = cmspf_emb_dim

        self.drug_extractor = MolecularGCN(
            in_feats=drug_in_feats,
            dim_embedding=drug_embedding,
            padding=drug_padding,
            hidden_feats=drug_hidden_feats,
        )
        self.protein_extractor = PMSAFI(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_module = CMSPF(
            nf=cmspf_emb_dim,
            num_heads=cmspf_num_head,
            dropout=cmspf_dropout_rate,
            num_cascades=cmspf_num_cascades,
        )

        self.drug_fusion = SCMAF(
            model_feat_dim=drug_hidden_feats[-1],
            precomputed_feat_dim=cfg_value(pretrained_cfg, "CHEMBERTA_DIM", 384),
            fusion_dim=drug_hidden_feats[-1],
            num_heads=cfg_value(drug_fusion_cfg, "NUM_HEAD", cmspf_num_head),
            dropout=cfg_value(drug_fusion_cfg, "DROPOUT_RATE", 0.1),
        )

        self.protein_fusion = MPMI(
            prott5_dim=cfg_value(pretrained_cfg, "PROTT5_DIM", 1024),
            esm2_dim=cfg_value(pretrained_cfg, "ESM2_DIM", 1280),
            fusion_dim=num_filters[-1],
            dropout=cfg_value(protein_fusion_cfg, "DROPOUT_RATE", 0.1),
            num_heads=cfg_value(protein_fusion_cfg, "NUM_HEAD", 4),
        )

        self.shared_feature_generator = nn.Sequential(
            nn.Linear(cmspf_emb_dim * 2, cmspf_emb_dim),
            nn.LayerNorm(cmspf_emb_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(cmspf_emb_dim, cmspf_emb_dim),
        )
        self.shared_gate = nn.Sequential(
            nn.Linear(cmspf_emb_dim * 2, cmspf_emb_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(cmspf_emb_dim * 3)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp_classifier = MPRC(
            mlp_in_dim,
            mlp_hidden_dim,
            mlp_out_dim,
            binary=out_binary,
            dropout_rate=mlp_dropout_rate,
        )

    def _pool_features(self, features):
        if features.dim() == 2:
            return features
        if features.dim() == 4:
            features = features.squeeze(-1)
        if features.dim() != 3:
            raise ValueError(f"Expected 2D/3D/4D features, got shape {tuple(features.shape)}")

        if features.shape[1] == self.feature_dim:
            channel_first = features
        elif features.shape[2] == self.feature_dim:
            channel_first = features.transpose(1, 2)
        else:
            channel_first = features if features.shape[1] < features.shape[2] else features.transpose(1, 2)

        return self.adaptive_avg_pool(channel_first).squeeze(-1)

    def forward(self, bg_d, v_p, drug_precomputed=None, protein_precomputed=None, mode="train"):
        v_d = self.drug_extractor(bg_d)      # [B, drug_len, D]
        v_p = self.protein_extractor(v_p)    # [B, D, protein_len, 1]

        if self.use_precomputed_features and drug_precomputed is not None and protein_precomputed is not None:
            if not isinstance(protein_precomputed, tuple) or len(protein_precomputed) != 2:
                raise ValueError("protein_precomputed must be a tuple: (esm2_features, prott5_features)")

            drug_precomputed = drug_precomputed.to(self.device)
            protein_esm2, protein_prott5 = protein_precomputed
            protein_esm2 = protein_esm2.to(self.device)
            protein_prott5 = protein_prott5.to(self.device)

            v_d_pooled = self._pool_features(v_d)
            protein_precomputed_proj = self.protein_fusion(protein_prott5, protein_esm2)
            if protein_precomputed_proj.dim() == 3:
                protein_precomputed_proj = self._pool_features(protein_precomputed_proj)

            v_d_fused = self.drug_fusion(v_d_pooled, drug_precomputed)
            v_p_fused = protein_precomputed_proj

            v_d = v_d_fused.unsqueeze(1).expand(-1, v_d.shape[1], -1)
            v_p = v_p_fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, v_p.shape[2], 1)

        min_seq_len = min(v_d.shape[1], v_p.shape[2])
        v_d = v_d[:, :min_seq_len, :]
        v_p = v_p[:, :, :min_seq_len, :]

        v_d = v_d.transpose(1, 2).unsqueeze(-1)

        v_d_flat = v_d.squeeze(-1).permute(0, 2, 1)
        v_p_flat = v_p.squeeze(-1).permute(0, 2, 1)
        v_concat = torch.cat([v_d_flat, v_p_flat], dim=-1)

        x_s = self.shared_feature_generator(v_concat) * self.shared_gate(v_concat)
        x_s = x_s.permute(0, 2, 1).unsqueeze(-1)

        v_d_enhanced, v_p_enhanced, x_s_enhanced = self.cross_module(v_d, v_p, x_s)

        v_d_pooled = self._pool_features(v_d_enhanced)
        v_p_pooled = self._pool_features(v_p_enhanced)
        x_s_pooled = self._pool_features(x_s_enhanced)

        f = torch.cat([v_d_pooled, v_p_pooled, x_s_pooled], dim=1)
        f = self.fusion_norm(f)
        score = self.mlp_classifier(f)

        if mode == "eval":
            return v_d_enhanced, v_p_enhanced, score, None
        if mode == "extract_features":
            return v_d_enhanced, v_p_enhanced, f, score
        return v_d_enhanced, v_p_enhanced, f, score
