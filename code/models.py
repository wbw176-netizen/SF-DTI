class SF_DTI(nn.Module):
    def __init__(self, device='cuda', use_precomputed_features=True, **config):
        super(SF_DTI, self).__init__()
        self.device = device
        self.use_precomputed_features = use_precomputed_features
        
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
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cmspf_num_head = config["SIGNAL"]['NUM_HEAD']
        cmspf_emb_dim = config['SIGNAL']['EMBEDDING_DIM']
        cmspf_dropout_rate = config['SIGNAL']['SIGNAL_DROPOUT_RATE']
        cmspf_num_cascades = config['SIGNAL']['NUM_CASCADES']
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = PMSAFI(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_module = CMSPF(nf=cmspf_emb_dim, num_heads=cmspf_num_head, dropout=cmspf_dropout_rate, num_cascades=cmspf_num_cascades)
        self.drug_fusion = SCMAF(
                model_feat_dim=drug_hidden_feats[-1],  
                precomputed_feat_dim=384, 
                fusion_dim=drug_hidden_feats[-1],
                dropout=0.1  
            )
            
        self.improved_protein_fusion = MSPMCA(
                prott5_dim=1024,
                esm2_dim=1280,  
                fusion_dim=num_filters[-1], 
                scales=[1, 2,4],  
                dropout=0.1, 
                num_heads=4,
            )
        self.shared_feature_generator = nn.Sequential(
            nn.Linear(cross_emb_dim * 2, cross_emb_dim),
            nn.LayerNorm(cross_emb_dim),
            nn.GELU(),
            nn.Dropout(0.05), 
            nn.Linear(cross_emb_dim, cross_emb_dim)
        )
        self.shared_gate = nn.Sequential(
            nn.Linear(cross_emb_dim * 2, cross_emb_dim),
            nn.Sigmoid() 
        )
        self.fusion_norm = nn.LayerNorm(cross_emb_dim * 3)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp_classifier = MPRC(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, dropout_rate=mlp_dropout_rate)
    
    def forward(self, bg_d, v_p, drug_precomputed=None, protein_precomputed=None, mode="train"):
