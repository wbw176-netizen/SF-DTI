import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
import math

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCEWithLogitsLoss()
    n = torch.sigmoid(torch.squeeze(pred_output, 1))
    loss = loss_fct(torch.squeeze(pred_output, 1), labels)
    return (n, loss)

def cross_entropy_logits(linear_output, label, weights=None):
    if linear_output.size(1) < 2:
        p = torch.sigmoid(linear_output)
        linear_output = torch.cat([1 - p, p], dim=1)
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction='none')(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return (n, loss)

def entropy_logits(linear_output):
    probs = F.softmax(linear_output, dim=1)
    log_probs = F.log_softmax(linear_output, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy.mean()

class SCMAF(nn.Module):

    def __init__(self, model_feat_dim, precomputed_feat_dim, fusion_dim, dropout=0.1):
        super().__init__()
        self.structure_path = nn.Sequential(nn.Linear(model_feat_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.property_path = nn.Sequential(nn.Linear(precomputed_feat_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.ot_alignment = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim * 2), nn.LayerNorm(fusion_dim * 2), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(fusion_dim * 2, fusion_dim))
        self.subspace_proj_gcn = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.subspace_proj_chem = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.subspace_basis = nn.Parameter(torch.randn(fusion_dim, fusion_dim) * 0.02)
        self.final_fusion = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim * 2), nn.LayerNorm(fusion_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(fusion_dim * 2, fusion_dim))
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
            if hasattr(torch, 'linalg'):
                q, _ = torch.linalg.qr(self.subspace_basis)
            else:
                q, _ = torch.qr(self.subspace_basis)
            self.subspace_basis.data = q

    def _optimal_transport_alignment(self, feat_a, feat_b):
        combined = torch.cat([feat_a, feat_b], dim=-1)
        return self.ot_alignment(combined)

    def _subspace_projection(self, feat, proj_type='gcn'):
        basis = F.normalize(self.subspace_basis, p=2, dim=1)
        projection = torch.matmul(basis, basis.T)
        feat_proj = self.subspace_proj_gcn(feat) if proj_type == 'gcn' else self.subspace_proj_chem(feat)
        return torch.matmul(feat_proj, projection.T)

    def forward(self, model_features, precomputed_features):
        f_gcn = self.structure_path(model_features)
        f_chem = self.property_path(precomputed_features)
        f_gcn_ot = self._optimal_transport_alignment(f_gcn, f_chem)
        f_chem_ot = self._optimal_transport_alignment(f_chem, f_gcn)
        f_gcn_proj = self._subspace_projection(f_gcn_ot, 'gcn')
        f_chem_proj = self._subspace_projection(f_chem_ot, 'chem')
        f_weighted = 0.5 * f_gcn_proj + 0.5 * f_chem_proj
        f_fused = self.final_fusion(torch.cat([f_gcn_proj, f_chem_proj], dim=-1))
        return self.layer_norm(f_fused + f_weighted + f_gcn + f_chem)

class MPMI(nn.Module):

    def __init__(self, prott5_dim=1024, esm2_dim=1280, fusion_dim=1024, dropout=0.1, num_heads=4):
        super().__init__()
        self.prott5_adapter = nn.Sequential(nn.LayerNorm(prott5_dim), nn.Linear(prott5_dim, fusion_dim * 2), nn.GELU(), nn.Linear(fusion_dim * 2, fusion_dim))
        self.esm2_adapter = nn.Sequential(nn.LayerNorm(esm2_dim), nn.Linear(esm2_dim, fusion_dim * 2), nn.GELU(), nn.Linear(fusion_dim * 2, fusion_dim))
        self.prott5_residual_compress = nn.Linear(prott5_dim, fusion_dim, bias=False)
        self.esm2_residual_compress = nn.Linear(esm2_dim, fusion_dim, bias=False)
        self.prott5_gate = nn.Parameter(torch.tensor(0.7))
        self.esm2_gate = nn.Parameter(torch.tensor(0.7))
        self.prott5_proj_norm = nn.LayerNorm(fusion_dim)
        self.esm2_proj_norm = nn.LayerNorm(fusion_dim)
        self.prott5_scale = nn.Parameter(torch.tensor(1.0))
        self.esm2_scale = nn.Parameter(torch.tensor(1.0))
        self.shared_interaction = SharedMutualInteraction(fusion_dim, num_heads, dropout)
        self.final_fusion = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(), nn.Dropout(dropout * 0.5))

    def forward(self, prott5_feat, esm2_feat):
        if prott5_feat.dim() != 2 or esm2_feat.dim() != 2:
            raise ValueError('MPMI expects pooled 2D protein features with shape [B, D].')
        return self._fuse_pooled_features(prott5_feat, esm2_feat)

    def _fuse_pooled_features(self, prott5_feat, esm2_feat):
        p_main = self.prott5_adapter(prott5_feat)
        e_main = self.esm2_adapter(esm2_feat)
        p_res = self.prott5_residual_compress(prott5_feat)
        e_res = self.esm2_residual_compress(esm2_feat)
        prott5_proj = self.prott5_proj_norm(self.prott5_gate * p_main + (1 - self.prott5_gate) * p_res)
        esm2_proj = self.esm2_proj_norm(self.esm2_gate * e_main + (1 - self.esm2_gate) * e_res)
        prott5_proj = self.prott5_scale * prott5_proj
        esm2_proj = self.esm2_scale * esm2_proj
        fused_feat = self.shared_interaction(prott5_proj, esm2_proj)
        return self.final_fusion(fused_feat)

class SharedMutualInteraction(nn.Module):

    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert feat_dim % num_heads == 0, 'feat_dim must be divisible by num_heads'
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
            mask = torch.empty(batch_size, self.num_heads, 1, device=v.device, dtype=v.dtype).bernoulli_(keep_prob).div_(keep_prob)
            v = v * mask
        return v.reshape(batch_size, self.feat_dim)

class CMSPF(nn.Module):

    def __init__(self, nf=128, dropout=0.1, num_cascades=4):
        super(CMSPF, self).__init__()
        self.nf = nf
        self.num_cascades = num_cascades
        self.dropout = nn.Dropout(dropout)
        self.semantic_nodes = nn.Parameter(torch.randn(nf, nf))
        self.node_bias = nn.Parameter(torch.zeros(nf, 1, 1))
        self.signal_weights = nn.Parameter(torch.randn(nf, nf))
        self.signal_bias = nn.Parameter(torch.zeros(nf, 1, 1))
        self.cross_modal_weights = nn.Parameter(torch.randn(2 * nf, nf))
        self.cross_modal_bias = nn.Parameter(torch.zeros(nf, 1, 1))
        self.energy_activation = nn.GELU()
        self.drug_transform = nn.ModuleList()
        self.protein_transform = nn.ModuleList()
        for i in range(num_cascades):
            if i == 0:
                self.drug_transform.append(self._make_lightweight_block(nf))
                self.protein_transform.append(self._make_lightweight_block(nf))
            elif i < num_cascades - 1:
                self.drug_transform.append(self._make_res_block(nf))
                self.protein_transform.append(self._make_res_block(nf))
            else:
                self.drug_transform.append(self._make_enhanced_block(nf))
                self.protein_transform.append(self._make_enhanced_block(nf))
        self.layer_attention = nn.ModuleList()
        for i in range(num_cascades):
            self.layer_attention.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(nf, nf // 4, 1), nn.GELU(), nn.Conv2d(nf // 4, 1, 1), nn.Sigmoid()))
        self.router = nn.ModuleList()
        for i in range(num_cascades):
            self.router.append(nn.ModuleDict({'gap': nn.AdaptiveAvgPool2d(1), 'cond_d_from_p': nn.Sequential(nn.Conv2d(nf, nf, 1), nn.GELU(), nn.Conv2d(nf, nf * 2, 1)), 'cond_p_from_d': nn.Sequential(nn.Conv2d(nf, nf, 1), nn.GELU(), nn.Conv2d(nf, nf * 2, 1)), 'route_d': nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1), nn.GELU()), 'route_p': nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1), nn.GELU()), 'spatial_mixer_d': nn.ModuleList([nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1)), nn.Sequential(nn.Conv2d(nf, nf, 3, padding=2, dilation=2, groups=nf), nn.Conv2d(nf, nf, 1)), nn.Sequential(nn.Conv2d(nf, nf, 3, padding=3, dilation=3, groups=nf), nn.Conv2d(nf, nf, 1))]), 'spatial_mixer_p': nn.ModuleList([nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1)), nn.Sequential(nn.Conv2d(nf, nf, 3, padding=2, dilation=2, groups=nf), nn.Conv2d(nf, nf, 1)), nn.Sequential(nn.Conv2d(nf, nf, 3, padding=3, dilation=3, groups=nf), nn.Conv2d(nf, nf, 1))]), 'fusion_d': nn.Sequential(nn.Conv2d(nf * 3, nf, 1), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1), nn.GELU(), nn.Conv2d(nf, nf, 1)), 'fusion_p': nn.Sequential(nn.Conv2d(nf * 3, nf, 1), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1), nn.GELU(), nn.Conv2d(nf, nf, 1)), 'ffn_d': nn.Sequential(nn.Conv2d(nf, nf * 4, 1), nn.GELU(), self.dropout, nn.Conv2d(nf * 4, nf, 1)), 'ffn_p': nn.Sequential(nn.Conv2d(nf, nf * 4, 1), nn.GELU(), self.dropout, nn.Conv2d(nf * 4, nf, 1)), 'se_d': nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(nf, nf // 8, 1), nn.GELU(), nn.Conv2d(nf // 8, nf, 1), nn.Sigmoid()), 'se_p': nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(nf, nf // 8, 1), nn.GELU(), nn.Conv2d(nf // 8, nf, 1), nn.Sigmoid()), 'shared_update': nn.Sequential(nn.Conv2d(nf * 3, nf, 1), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1)), 'bn_shared': nn.BatchNorm2d(nf)}))
        self.ms_fuse = nn.ModuleDict({'s1': nn.Conv2d(nf, nf, 1), 's2': nn.Conv2d(nf, nf, 1), 's3': nn.Conv2d(nf, nf, 1), 'combine': nn.Conv2d(nf * 3, nf, 1)})
        self.final_fusion = nn.Sequential(nn.Conv2d(nf * 3, nf, 1), nn.BatchNorm2d(nf), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1), nn.GELU(), nn.Conv2d(nf, nf, 1))
        self._init_weights()

    def _make_res_block(self, nf):
        return nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf))

    def _make_lightweight_block(self, nf):
        return nn.Sequential(nn.Conv2d(nf, nf, 1), nn.BatchNorm2d(nf), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1, groups=nf), nn.Conv2d(nf, nf, 1), nn.BatchNorm2d(nf))

    def _make_enhanced_block(self, nf):
        return nn.Sequential(nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.GELU(), nn.Conv2d(nf, nf, 3, padding=1), nn.BatchNorm2d(nf), nn.GELU(), nn.Conv2d(nf, nf, 1), nn.BatchNorm2d(nf))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.semantic_nodes)
        nn.init.xavier_uniform_(self.signal_weights)
        nn.init.xavier_uniform_(self.cross_modal_weights)

    def _signal_propagation(self, x_drug, x_protein):
        B, C, H, W = x_drug.shape
        drug_semantic = torch.matmul(x_drug.view(B, C, -1).transpose(1, 2), self.semantic_nodes)
        protein_semantic = torch.matmul(x_protein.view(B, C, -1).transpose(1, 2), self.semantic_nodes)
        drug_semantic = drug_semantic + self.node_bias.squeeze(-1).transpose(0, 1)
        protein_semantic = protein_semantic + self.node_bias.squeeze(-1).transpose(0, 1)
        drug_signal = self.energy_activation(torch.matmul(drug_semantic, self.signal_weights.transpose(0, 1)) + self.signal_bias.squeeze(-1).transpose(0, 1))
        protein_signal = self.energy_activation(torch.matmul(protein_semantic, self.signal_weights.transpose(0, 1)) + self.signal_bias.squeeze(-1).transpose(0, 1))
        drug_signal = drug_signal.transpose(1, 2).unsqueeze(-1)
        drug_signal = drug_signal.view(B, self.nf, H, W)
        protein_signal = protein_signal.transpose(1, 2).unsqueeze(-1)
        protein_signal = protein_signal.view(B, self.nf, H, W)
        cross_modal_signal = self.energy_activation(torch.matmul(torch.cat([drug_semantic, protein_semantic], dim=-1), self.cross_modal_weights) + self.cross_modal_bias.squeeze(-1).transpose(0, 1))
        cross_modal_signal = cross_modal_signal.transpose(1, 2).unsqueeze(-1)
        cross_modal_signal = cross_modal_signal.view(B, self.nf, H, W)
        return (drug_signal, protein_signal, cross_modal_signal)

    def _signal_routing(self, x_drug, x_protein, drug_signal, protein_signal, cross_modal_signal):
        drug_energy_map = torch.norm(drug_signal, dim=1, keepdim=True)
        protein_energy_map = torch.norm(protein_signal, dim=1, keepdim=True)
        cross_energy_map = torch.norm(cross_modal_signal, dim=1, keepdim=True)
        total_energy = drug_energy_map + protein_energy_map + cross_energy_map + 1e-08
        drug_weight = drug_energy_map / total_energy
        protein_weight = protein_energy_map / total_energy
        cross_weight = cross_energy_map / total_energy
        if x_drug.shape[1] != self.nf:
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
        return (routed_drug, routed_protein, cross_modal_signal)

    def _ms_features(self, x):
        s1 = self.ms_fuse['s1'](x)
        h, w = (x.shape[2], x.shape[3])
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
        x_drug_res, x_protein_res, x_shared_res = (x_drug, x_protein, x_shared)
        drug_signal, protein_signal, cross_modal_signal = self._signal_propagation(x_drug, x_protein)
        for i in range(self.num_cascades):
            trans_d = self.drug_transform[i](x_drug)
            trans_p = self.protein_transform[i](x_protein)
            rt = self.router[i]
            routed_d, routed_p, cross_modal_signal = self._signal_routing(trans_d, trans_p, drug_signal, protein_signal, cross_modal_signal)
            gd = rt['gap'](routed_d)
            gp = rt['gap'](routed_p)
            cond_d = rt['cond_d_from_p'](gp)
            cond_p = rt['cond_p_from_d'](gd)
            scale_d, shift_d = torch.chunk(cond_d, 2, dim=1)
            scale_p, shift_p = torch.chunk(cond_p, 2, dim=1)
            mod_d = routed_d * torch.sigmoid(scale_d) + shift_d
            mod_p = routed_p * torch.sigmoid(scale_p) + shift_p
            routed_d = rt['route_d'](mod_p)
            routed_p = rt['route_p'](mod_d)
            ms_d = sum((branch(mod_d) for branch in rt['spatial_mixer_d'])) / 3.0
            ms_p = sum((branch(mod_p) for branch in rt['spatial_mixer_p'])) / 3.0
            attn_d = rt['se_d'](ms_d)
            attn_p = rt['se_p'](ms_p)
            ms_d = ms_d * attn_d
            ms_p = ms_p * attn_p
            fused_d = rt['fusion_d'](torch.cat([trans_d, routed_d, ms_d], dim=1))
            fused_p = rt['fusion_p'](torch.cat([trans_p, routed_p, ms_p], dim=1))
            ffn_d = rt['ffn_d'](fused_d) + fused_d
            ffn_p = rt['ffn_p'](fused_p) + fused_p
            layer_attn = self.layer_attention[i](ffn_d + ffn_p)
            ffn_d = ffn_d * layer_attn
            ffn_p = ffn_p * layer_attn
            shared_input = torch.cat([ffn_d, ffn_p, cross_modal_signal], dim=1)
            x_shared = rt['bn_shared'](rt['shared_update'](shared_input)) + x_shared
            x_drug = ffn_d + x_drug
            x_protein = ffn_p + x_protein
            if i < self.num_cascades - 1:
                drug_signal, protein_signal, cross_modal_signal = self._signal_propagation(x_drug, x_protein)
        ms_d = self._ms_features(x_drug)
        ms_p = self._ms_features(x_protein)
        final_shared = self.final_fusion(torch.cat([ms_d, ms_p, x_shared], dim=1))
        return (x_drug + x_drug_res, x_protein + x_protein_res, final_shared + x_shared_res)

class MolecularGCN(nn.Module):

    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation, allow_zero_in_degree=True)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        batch_num_nodes = batch_graph.batch_num_nodes()
        max_nodes = max(batch_num_nodes).item()
        output = torch.zeros(batch_size, max_nodes, self.output_feats, device=node_feats.device)
        start_idx = 0
        for i in range(batch_size):
            num_nodes = batch_num_nodes[i].item()
            output[i, :num_nodes, :] = node_feats[start_idx:start_idx + num_nodes, :]
            start_idx += num_nodes
        return output

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32', 'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32', 'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return (mapper_x, mapper_y)

class NonlinearFrequencyTransform(nn.Module):

    def __init__(self, in_channels, transform_type='swish', num_bases=4):
        super(NonlinearFrequencyTransform, self).__init__()
        self.transform_type = transform_type
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.frequency_bases = nn.Parameter(torch.randn(num_bases, in_channels) * 0.1)
        self.basis_weights = nn.Parameter(torch.ones(num_bases))
        self.nonlinear_activations = nn.ModuleDict({'swish': nn.SiLU(), 'gelu': nn.GELU(), 'relu': nn.ReLU(inplace=True)})
        self.channel_reconstructors = {}

    def polynomial_transform(self, x, degree=2):
        return x + self.alpha ** 2 * x ** 2

    def adaptive_basis_transform(self, x):
        bases = self.frequency_bases
        weights = F.softmax(self.basis_weights, dim=0)
        x_reshaped = x.view(x.size(0), x.size(1), -1)
        similarities = torch.einsum('nc,bch->nbh', bases, x_reshaped)
        weighted_response = torch.einsum('n,nbh->bh', weights, similarities)
        weighted_response_reshaped = weighted_response.view(x.size(0), 1, x.size(2), x.size(3))
        channels = x.size(1)
        if channels not in self.channel_reconstructors:
            self.channel_reconstructors[channels] = nn.Conv2d(1, channels, 1, bias=False).to(x.device)
        return self.channel_reconstructors[channels](weighted_response_reshaped)

    def forward(self, x, transform_method=None):
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
            return self.nonlinear_activations['swish'](x)

class EnhancedMultiFrequencyChannelAttention(nn.Module):

    def __init__(self, in_channels, dct_h=7, dct_w=7, frequency_branches=8, frequency_selection='top', reduction=16, use_nonlinear=True):
        super(EnhancedMultiFrequencyChannelAttention, self).__init__()
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        frequency_selection = frequency_selection + str(frequency_branches)
        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.use_nonlinear = use_nonlinear
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        if use_nonlinear:
            self.nonlinear_transform = NonlinearFrequencyTransform(in_channels, transform_type='swish', num_bases=4)
            self.transform_selector = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 3, 1), nn.Softmax(dim=1))
        self.register_buffer('dct_weights', self._precompute_dct_weights(mapper_x, mapper_y, in_channels))
        self.fc = nn.Sequential(nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.freq_response_learner = nn.Sequential(nn.Conv2d(in_channels, in_channels // 8, 1), nn.ReLU(inplace=True), nn.Conv2d(in_channels // 8, in_channels, 1), nn.Sigmoid())

    def _precompute_dct_weights(self, mapper_x, mapper_y, in_channels):
        dct_weights = []
        for freq_idx in range(self.num_freq):
            dct_weight = torch.zeros(in_channels, in_channels, self.dct_h, self.dct_w)
            for t_x in range(self.dct_h):
                for t_y in range(self.dct_w):
                    for ch in range(in_channels):
                        dct_weight[ch, ch, t_x, t_y] = self._build_filter(t_x, mapper_x[freq_idx], self.dct_h) * self._build_filter(t_y, mapper_y[freq_idx], self.dct_w)
            dct_weights.append(dct_weight)
        return torch.stack(dct_weights, dim=0)

    def _build_filter(self, pos, freq, size):
        result = math.cos(math.pi * freq * (2 * pos + 1) / (2 * size)) / math.sqrt(size)
        if freq == 0:
            result = result * math.sqrt(2)
        return result

    def forward(self, x):
        b, c, h, w = x.size()
        freq_features = []
        for freq_idx in range(self.num_freq):
            dct_weight = self.dct_weights[freq_idx]
            freq_feat = F.conv2d(x, dct_weight, padding=(self.dct_h // 2, self.dct_w // 2), stride=1)
            if self.use_nonlinear:
                transform_weights = self.transform_selector(freq_feat)
                transform_methods = ['swish', 'gelu', 'adaptive']
                nonlinear_features = []
                for i, method in enumerate(transform_methods):
                    transformed = self.nonlinear_transform(freq_feat, method)
                    weighted = transformed * transform_weights[:, i:i + 1, :, :]
                    nonlinear_features.append(weighted)
                freq_feat = sum(nonlinear_features)
            freq_features.append(freq_feat)
        freq_fused = torch.stack(freq_features, dim=1).mean(dim=1)
        freq_response = self.freq_response_learner(freq_fused)
        freq_fused = freq_fused * freq_response
        avg_out = self.fc(self.avg_pool(freq_fused))
        max_out = self.fc(self.max_pool(freq_fused))
        attention = torch.sigmoid(avg_out + max_out)
        out = x * attention
        enhanced = torch.cat([out, freq_fused], dim=1)
        out = self.fusion_conv(enhanced)
        return out

class PMSAFI(nn.Module):

    def __init__(self, embedding_dim, num_filters, padding=True):
        super(PMSAFI, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.mfca = EnhancedMultiFrequencyChannelAttention(in_channels=in_ch[0], frequency_branches=8, frequency_selection='top', reduction=16, use_nonlinear=True)
        self.feature_align = nn.Sequential(nn.Conv2d(in_ch[0], in_ch[1], 1), nn.BatchNorm2d(in_ch[1]), nn.ReLU())
        self.sequence_modeling = nn.Sequential(nn.Conv1d(in_ch[0], in_ch[1], kernel_size=3, padding=1), nn.BatchNorm1d(in_ch[1]), nn.ReLU(), nn.Conv1d(in_ch[1], in_ch[1], kernel_size=5, padding=2), nn.BatchNorm1d(in_ch[1]), nn.ReLU())

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = v.unsqueeze(-1)
        v = self.mfca(v)
        v = v.squeeze(-1)
        v = self.sequence_modeling(v)
        v = v.unsqueeze(-1)
        v = self.feature_align(v)
        return v

class MPRC(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout_rate=None):
        super(MPRC, self).__init__()
        self.proj1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.proj2 = nn.Linear(in_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.fuse = nn.Linear(in_dim * 3, hidden_dim)
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        self.hidden_fc = nn.Linear(hidden_dim, out_dim)
        self.bn_hidden = nn.BatchNorm1d(out_dim)
        self.output_fc = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_identity = x
        x_axis1 = self.relu(self.bn1(self.proj1(x)))
        x_axis2 = self.relu(self.bn2(self.proj2(x)))
        x_concat = torch.cat([x_axis1, x_axis2, x_identity], dim=1)
        x_fused = self.relu(self.bn_fuse(self.fuse(x_concat)))
        x_hidden = self.dropout(self.relu(self.bn_hidden(self.hidden_fc(x_fused))))
        output = self.output_fc(x_hidden)
        return output

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
        signal_cfg = cfg_value(config, 'SIGNAL')
        if signal_cfg is None:
            signal_cfg = cfg_value(config, 'CBIE_CROSS_ATTENTION')
        if signal_cfg is None:
            raise KeyError('Missing SIGNAL config section')
        pretrained_cfg = cfg_value(config, 'PRETRAINED', {})
        drug_fusion_cfg = cfg_value(config, 'DRUG_FUSION', {})
        protein_fusion_cfg = cfg_value(config, 'PROTEIN_FUSION', {})
        drug_in_feats = config['DRUG']['NODE_IN_FEATS']
        drug_embedding = config['DRUG']['NODE_IN_EMBEDDING']
        drug_hidden_feats = config['DRUG']['HIDDEN_LAYERS']
        protein_emb_dim = config['PROTEIN']['EMBEDDING_DIM']
        num_filters = config['PROTEIN']['NUM_FILTERS']
        mlp_in_dim = config['DECODER']['IN_DIM']
        mlp_hidden_dim = config['DECODER']['HIDDEN_DIM']
        mlp_out_dim = config['DECODER']['OUT_DIM']
        mlp_dropout_rate = config['DECODER']['DROPOUT_RATE']
        drug_padding = config['DRUG']['PADDING']
        protein_padding = config['PROTEIN']['PADDING']
        out_binary = config['DECODER']['BINARY']
        cmspf_emb_dim = signal_cfg['EMBEDDING_DIM']
        cmspf_num_cascades = signal_cfg['NUM_CASCADES']
        cmspf_dropout_rate = cfg_value(signal_cfg, 'SIGNAL_DROPOUT_RATE', cfg_value(signal_cfg, 'CBIE_DROPOUT_RATE', 0.1))
        self.feature_dim = cmspf_emb_dim
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding, padding=drug_padding, hidden_feats=drug_hidden_feats)
        self.protein_extractor = PMSAFI(protein_emb_dim, num_filters, protein_padding)
        self.cross_module = CMSPF(nf=cmspf_emb_dim, dropout=cmspf_dropout_rate, num_cascades=cmspf_num_cascades)
        self.drug_fusion = SCMAF(model_feat_dim=drug_hidden_feats[-1], precomputed_feat_dim=cfg_value(pretrained_cfg, 'CHEMBERTA_DIM', 384), fusion_dim=drug_hidden_feats[-1], dropout=cfg_value(drug_fusion_cfg, 'DROPOUT_RATE', 0.1))
        self.protein_fusion = MPMI(prott5_dim=cfg_value(pretrained_cfg, 'PROTT5_DIM', 1024), esm2_dim=cfg_value(pretrained_cfg, 'ESM2_DIM', 1280), fusion_dim=num_filters[-1], dropout=cfg_value(protein_fusion_cfg, 'DROPOUT_RATE', 0.1), num_heads=cfg_value(protein_fusion_cfg, 'NUM_HEAD', 4))
        self.shared_feature_generator = nn.Sequential(nn.Linear(cmspf_emb_dim * 2, cmspf_emb_dim), nn.LayerNorm(cmspf_emb_dim), nn.GELU(), nn.Dropout(0.05), nn.Linear(cmspf_emb_dim, cmspf_emb_dim))
        self.shared_gate = nn.Sequential(nn.Linear(cmspf_emb_dim * 2, cmspf_emb_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(cmspf_emb_dim * 3)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp_classifier = MPRC(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary, dropout_rate=mlp_dropout_rate)

    def _pool_features(self, features):
        if features.dim() == 2:
            return features
        if features.dim() == 4:
            features = features.squeeze(-1)
        if features.dim() != 3:
            raise ValueError(f'Expected 2D/3D/4D features, got shape {tuple(features.shape)}')
        if features.shape[1] == self.feature_dim:
            channel_first = features
        elif features.shape[2] == self.feature_dim:
            channel_first = features.transpose(1, 2)
        else:
            channel_first = features if features.shape[1] < features.shape[2] else features.transpose(1, 2)
        return self.adaptive_avg_pool(channel_first).squeeze(-1)

    def forward(self, bg_d, v_p, drug_precomputed=None, protein_precomputed=None, mode='train'):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        if self.use_precomputed_features and drug_precomputed is not None and (protein_precomputed is not None):
            if not isinstance(protein_precomputed, tuple) or len(protein_precomputed) != 2:
                raise ValueError('protein_precomputed must be a tuple: (esm2_features, prott5_features)')
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
        if mode == 'eval':
            return (v_d_enhanced, v_p_enhanced, score, None)
        if mode == 'extract_features':
            return (v_d_enhanced, v_p_enhanced, f, score)
        return (v_d_enhanced, v_p_enhanced, f, score)
