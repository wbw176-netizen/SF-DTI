import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN

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
        num_filters = config['PROTEIN']['NUM_FILTERS']
        mlp_in_dim = config['DECODER']['IN_DIM']
        mlp_hidden_dim = config['DECODER']['HIDDEN_DIM']
        mlp_out_dim = config['DECODER']['OUT_DIM']
        mlp_dropout_rate = config['DECODER']['DROPOUT_RATE']
        drug_padding = config['DRUG']['PADDING']
        out_binary = config['DECODER']['BINARY']
        cmspf_emb_dim = signal_cfg['EMBEDDING_DIM']
        cmspf_num_cascades = signal_cfg['NUM_CASCADES']
        cmspf_dropout_rate = cfg_value(signal_cfg, 'SIGNAL_DROPOUT_RATE', cfg_value(signal_cfg, 'CBIE_DROPOUT_RATE', 0.1))
        self.feature_dim = cmspf_emb_dim
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding, padding=drug_padding, hidden_feats=drug_hidden_feats)
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
        protein_seq_len = v_p.shape[1]
        if not self.use_precomputed_features:
            raise ValueError('SF_DTI requires precomputed ChemBERTa, ESM-2, and ProtT5 features')
        if drug_precomputed is None or protein_precomputed is None:
            raise ValueError('Precomputed drug and protein features are required')
        if not isinstance(protein_precomputed, tuple) or len(protein_precomputed) != 2:
            raise ValueError('protein_precomputed must be a tuple: (esm2_features, prott5_features)')
        drug_precomputed = drug_precomputed.to(self.device)
        protein_esm2, protein_prott5 = protein_precomputed
        protein_esm2 = protein_esm2.to(self.device)
        protein_prott5 = protein_prott5.to(self.device)
        v_d_pooled = self._pool_features(v_d)
        v_p_fused = self.protein_fusion(protein_prott5, protein_esm2)
        v_d_fused = self.drug_fusion(v_d_pooled, drug_precomputed)
        v_d = v_d_fused.unsqueeze(1).expand(-1, v_d.shape[1], -1)
        v_p = v_p_fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, protein_seq_len, 1)
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
