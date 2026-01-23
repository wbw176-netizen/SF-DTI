import torch.utils.data as data
import torch
import numpy as np
import os
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein

class DTIDataset(data.Dataset):

    def __init__(self, list_IDs,  df, max_drug_nodes=300, precomputed_features_dir=None):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes
        self.precomputed_features_dir = precomputed_features_dir

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)
        
        # 检查预训练特征是否存在
        self.has_precomputed_features = False
        if precomputed_features_dir and os.path.exists(precomputed_features_dir):
            self.has_precomputed_features = True
            self._load_precomputed_features()
    
    def _load_precomputed_features(self):
        """加载预训练特征"""
        try:
            # 尝试不同的文件名模式
            possible_smiles_files = [
                'smiles_features.npy',
                'train_smiles_features.npy',
                'val_smiles_features.npy', 
                'test_smiles_features.npy'
            ]
            
            # 分别查找ESM2和ProtT5特征文件
            possible_esm2_files = [
                'protein_features_esm2.npy',
                'train_protein_features_esm2.npy',
                'val_protein_features_esm2.npy',
                'test_protein_features_esm2.npy'
            ]
            
            possible_prott5_files = [
                'protein_features_prott5.npy',
                'train_protein_features_prott5.npy',
                'val_protein_features_prott5.npy',
                'test_protein_features_prott5.npy'
            ]
            
            # 查找存在的文件
            smiles_features_path = None
            esm2_features_path = None
            prott5_features_path = None
            
            for filename in possible_smiles_files:
                path = os.path.join(self.precomputed_features_dir, filename)
                if os.path.exists(path):
                    smiles_features_path = path
                    break
            
            for filename in possible_esm2_files:
                path = os.path.join(self.precomputed_features_dir, filename)
                if os.path.exists(path):
                    esm2_features_path = path
                    break
            
            for filename in possible_prott5_files:
                path = os.path.join(self.precomputed_features_dir, filename)
                if os.path.exists(path):
                    prott5_features_path = path
                    break
            
            if smiles_features_path and esm2_features_path and prott5_features_path:
                self.smiles_features = np.load(smiles_features_path)
                self.esm2_features = np.load(esm2_features_path)
                self.prott5_features = np.load(prott5_features_path)
                print(f"成功加载预训练特征:")
                print(f"  SMILES: {os.path.basename(smiles_features_path)} {self.smiles_features.shape}")
                print(f"  ESM2: {os.path.basename(esm2_features_path)} {self.esm2_features.shape}")
                print(f"  ProtT5: {os.path.basename(prott5_features_path)} {self.prott5_features.shape}")
            else:
                print("预训练特征文件不存在，将使用原始特征")
                print(f"查找目录: {self.precomputed_features_dir}")
                print(f"尝试的文件名: {possible_smiles_files + possible_esm2_files + possible_prott5_files}")
                self.has_precomputed_features = False
        except Exception as e:
            print(f"加载预训练特征失败: {e}")
            self.has_precomputed_features = False

    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)

        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]

        # Add a safety check to ensure num_virtual_nodes is non-negative
        num_virtual_nodes = max(0, self.max_drug_nodes - num_actual_nodes)

        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats

        # Only add virtual nodes if num_virtual_nodes > 0
        if num_virtual_nodes > 0:
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                           torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})

        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]['Y']

        # 预训练特征
        drug_precomputed = None
        protein_precomputed = None
        
        if self.has_precomputed_features:
            drug_precomputed = torch.tensor(self.smiles_features[index], dtype=torch.float32)
            # 返回ESM2和ProtT5特征的元组
            esm2_feat = torch.tensor(self.esm2_features[index], dtype=torch.float32)
            prott5_feat = torch.tensor(self.prott5_features[index], dtype=torch.float32)
            protein_precomputed = (esm2_feat, prott5_feat)

        return v_d, v_p, y, drug_precomputed, protein_precomputed

class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
