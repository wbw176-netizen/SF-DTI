#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import EsmTokenizer, EsmModel
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging
import itertools
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiPoolingFeatureExtractor:
    
    def __init__(self, 
                 protein_model_name: str = "facebook/esm2_t33_650M_UR50D",
                 smiles_model_name: str = "DeepChem/ChemBERTa-77M-MLM"):
        
        self.protein_model_name = protein_model_name
        self.smiles_model_name = smiles_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.pooling_strategies = {
            "cls": "仅使用[CLS]标记",
            "mean": "平均池化",
            "max": "最大池化", 
            "cls_mean": "[CLS] + 平均池化",
            "cls_max": "[CLS] + 最大池化",
            "mean_max": "平均池化 + 最大池化",
            "all": "所有池化策略组合"
        }
        
        logger.info(f"初始化多池化特征提取器")
        logger.info(f"使用设备: {self.device}")
        logger.info(f"支持的池化策略: {list(self.pooling_strategies.keys())}")
    
    def _load_models(self):
        """加载模型"""
        logger.info("加载ESM2模型...")
        self.protein_tokenizer = EsmTokenizer.from_pretrained(self.protein_model_name)
        self.protein_model = EsmModel.from_pretrained(self.protein_model_name)
        self.protein_model.to(self.device)
        self.protein_model.eval()
        self.protein_dim = self.protein_model.config.hidden_size
        
        logger.info("加载ChemBERTa模型...")
        self.smiles_tokenizer = AutoTokenizer.from_pretrained(self.smiles_model_name)
        self.smiles_model = AutoModel.from_pretrained(self.smiles_model_name)
        self.smiles_model.to(self.device)
        self.smiles_model.eval()
        self.smiles_dim = self.smiles_model.config.hidden_size
        
        logger.info(f"ESM2特征维度: {self.protein_dim}")
        logger.info(f"ChemBERTa特征维度: {self.smiles_dim}")
    
    def _pool_features(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, 
                      strategy: str) -> torch.Tensor:
        """多种池化策略实现"""
        
        if strategy == "cls":
            # 仅使用[CLS]标记
            return hidden_states[:, 0, :]
        
        elif strategy == "mean":
            # 平均池化
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif strategy == "max":
            # 最大池化
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            return torch.max(hidden_states, 1)[0]
        
        elif strategy == "cls_mean":
            # [CLS] + 平均池化
            cls_features = hidden_states[:, 0, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_features = sum_embeddings / sum_mask
            return (cls_features + mean_features) / 2
        
        elif strategy == "cls_max":
            # [CLS] + 最大池化
            cls_features = hidden_states[:, 0, :]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[input_mask_expanded == 0] = -1e9
            max_features = torch.max(hidden_states, 1)[0]
            return (cls_features + max_features) / 2
        
        elif strategy == "mean_max":
            # 平均池化 + 最大池化
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_features = sum_embeddings / sum_mask
            
            hidden_states[input_mask_expanded == 0] = -1e9
            max_features = torch.max(hidden_states, 1)[0]
            return (mean_features + max_features) / 2
        
        elif strategy == "all":
            # 所有池化策略组合
            cls_features = hidden_states[:, 0, :]
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_features = sum_embeddings / sum_mask
            
            hidden_states[input_mask_expanded == 0] = -1e9
            max_features = torch.max(hidden_states, 1)[0]
            
            # 组合所有特征
            return torch.cat([cls_features, mean_features, max_features], dim=1)
        
        else:
            raise ValueError(f"不支持的池化策略: {strategy}")
    
    def extract_protein_features(self, protein_sequences: List[str], 
                               pooling_strategy: str, batch_size: int = 8) -> np.ndarray:
        """提取蛋白质特征"""
        logger.info(f"提取蛋白质特征 - 策略: {pooling_strategy}")
        
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(protein_sequences), batch_size), 
                         desc=f"蛋白质特征-{pooling_strategy}"):
                batch_sequences = protein_sequences[i:i + batch_size]
                
                # 分词
                batch_encoded = self.protein_tokenizer(
                    batch_sequences,
                    padding=True,
                    truncation=True,
                    max_length=2048,  # 增加最大长度
                    return_tensors="pt"
                )
                
                # 移动到设备
                batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
                
                # 前向传播
                outputs = self.protein_model(**batch_encoded)
                
                # 池化
                batch_features = self._pool_features(
                    outputs.last_hidden_state, 
                    batch_encoded['attention_mask'],
                    pooling_strategy
                ).cpu().numpy()
                
                all_features.append(batch_features)
        
        features = np.vstack(all_features)
        logger.info(f"蛋白质特征提取完成 - 形状: {features.shape}")
        return features
    
    def extract_smiles_features(self, smiles_list: List[str], 
                              pooling_strategy: str, batch_size: int = 16) -> np.ndarray:
        """提取SMILES特征"""
        logger.info(f"提取SMILES特征 - 策略: {pooling_strategy}")
        
        all_features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(smiles_list), batch_size), 
                         desc=f"SMILES特征-{pooling_strategy}"):
                batch_smiles = smiles_list[i:i + batch_size]
                
                # 分词
                batch_encoded = self.smiles_tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=1024,  # 增加最大长度
                    return_tensors="pt"
                )
                
                # 移动到设备
                batch_encoded = {k: v.to(self.device) for k, v in batch_encoded.items()}
                
                # 前向传播
                outputs = self.smiles_model(**batch_encoded)
                
                # 池化
                batch_features = self._pool_features(
                    outputs.last_hidden_state, 
                    batch_encoded['attention_mask'],
                    pooling_strategy
                ).cpu().numpy()
                
                all_features.append(batch_features)
        
        features = np.vstack(all_features)
        logger.info(f"SMILES特征提取完成 - 形状: {features.shape}")
        return features
    
    def extract_features_with_strategy(self, csv_path: str, 
                                     protein_pooling: str, smiles_pooling: str,
                                     output_dir: str, batch_size: int = 8) -> Dict[str, np.ndarray]:
        """使用指定策略提取特征"""
        
        # 读取数据
        df = pd.read_csv(csv_path)
        smiles_list = df['SMILES'].tolist()
        protein_list = df['Protein'].tolist()
        labels = df['Y'].values if 'Y' in df.columns else None
        
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"SMILES数量: {len(smiles_list)}")
        logger.info(f"蛋白质数量: {len(protein_list)}")
        
        # 提取特征
        smiles_features = self.extract_smiles_features(smiles_list, smiles_pooling, batch_size)
        protein_features = self.extract_protein_features(protein_list, protein_pooling, batch_size)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名
        file_name = os.path.splitext(os.path.basename(csv_path))[0]
        strategy_name = f"{protein_pooling}_{smiles_pooling}"
        
        # 保存特征
        features_dict = {
            'smiles_features': smiles_features,
            'protein_features': protein_features,
            'labels': labels,
            'smiles_list': smiles_list,
            'protein_list': protein_list,
            'strategy': strategy_name
        }
        
        # 保存到文件
        np.save(os.path.join(output_dir, f"{file_name}_smiles_{strategy_name}.npy"), smiles_features)
        np.save(os.path.join(output_dir, f"{file_name}_protein_{strategy_name}.npy"), protein_features)
        
        if labels is not None:
            np.save(os.path.join(output_dir, f"{file_name}_labels_{strategy_name}.npy"), labels)
        
        # 保存元数据
        metadata = {
            'file_name': file_name,
            'strategy': strategy_name,
            'protein_pooling': protein_pooling,
            'smiles_pooling': smiles_pooling,
            'total_samples': len(smiles_list),
            'smiles_feature_dim': smiles_features.shape[1],
            'protein_feature_dim': protein_features.shape[1],
            'model_info': {
                'smiles_model': self.smiles_model_name,
                'protein_model': self.protein_model_name
            },
            'extraction_time': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(output_dir, f"{file_name}_metadata_{strategy_name}.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"特征保存完成 - 策略: {strategy_name}")
        logger.info(f"SMILES特征形状: {smiles_features.shape}")
        logger.info(f"蛋白质特征形状: {protein_features.shape}")
        
        return features_dict
    
    def batch_extract_all_strategies(self, csv_path: str, 
                                   output_base_dir: str = "features_pooling_comparison",
                                   batch_size: int = 8,
                                   strategies_to_test: Optional[List[str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """批量提取所有池化策略组合的特征"""
        
        if strategies_to_test is None:
            strategies_to_test = ["cls", "mean", "max", "cls_mean"]
        
        logger.info(f"开始批量提取特征 - 测试策略: {strategies_to_test}")
        logger.info(f"CSV文件: {csv_path}")
        logger.info(f"输出目录: {output_base_dir}")
        
        # 加载模型
        self._load_models()
        
        # 生成所有策略组合
        strategy_combinations = list(itertools.product(strategies_to_test, strategies_to_test))
        logger.info(f"总共 {len(strategy_combinations)} 种策略组合")
        
        all_results = {}
        
        for i, (protein_pooling, smiles_pooling) in enumerate(strategy_combinations):
            logger.info(f"\n{'='*60}")
            logger.info(f"处理组合 {i+1}/{len(strategy_combinations)}: {protein_pooling} + {smiles_pooling}")
            logger.info(f"{'='*60}")
            
            try:
                # 创建策略特定的输出目录
                strategy_output_dir = os.path.join(output_base_dir, f"{protein_pooling}_{smiles_pooling}")
                
                # 提取特征
                start_time = time.time()
                features = self.extract_features_with_strategy(
                    csv_path=csv_path,
                    protein_pooling=protein_pooling,
                    smiles_pooling=smiles_pooling,
                    output_dir=strategy_output_dir,
                    batch_size=batch_size
                )
                
                extraction_time = time.time() - start_time
                logger.info(f"✓ 完成 - 耗时: {extraction_time:.2f}秒")
                
                all_results[f"{protein_pooling}_{smiles_pooling}"] = features
                
            except Exception as e:
                logger.error(f"✗ 失败: {e}")
                continue
        
        # 保存总结报告
        self._save_comparison_report(all_results, output_base_dir)
        
        logger.info(f"\n{'='*60}")
        logger.info("批量特征提取完成")
        logger.info(f"{'='*60}")
        logger.info(f"成功完成: {len(all_results)} 种策略组合")
        logger.info(f"输出目录: {output_base_dir}")
        
        return all_results
    
    def _save_comparison_report(self, all_results: Dict[str, Dict[str, np.ndarray]], 
                              output_base_dir: str):
        """保存比较报告"""
        
        report = {
            'extraction_summary': {
                'total_strategies': len(all_results),
                'extraction_time': pd.Timestamp.now().isoformat()
            },
            'strategy_results': {}
        }
        
        for strategy_name, features in all_results.items():
            report['strategy_results'][strategy_name] = {
                'smiles_feature_shape': features['smiles_features'].shape,
                'protein_feature_shape': features['protein_features'].shape,
                'total_samples': features['smiles_features'].shape[0],
                'smiles_feature_dim': features['smiles_features'].shape[1],
                'protein_feature_dim': features['protein_features'].shape[1]
            }
        
        # 保存报告
        report_path = os.path.join(output_base_dir, "pooling_strategy_comparison_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"比较报告已保存: {report_path}")

def main():
    """主函数 - 批量测试所有池化策略"""
    
    # 初始化批量提取器
    extractor = MultiPoolingFeatureExtractor()
    
    # 测试文件路径
    test_csv = "../datasets/Drugbank/random/train.csv"
    
    if not os.path.exists(test_csv):
        logger.error(f"测试文件不存在: {test_csv}")
        return
    
    # 批量提取所有策略组合
    results = extractor.batch_extract_all_strategies(
        csv_path=test_csv,
        output_base_dir="../code/features/pooling_strategy_comparison",
        batch_size=4,  # 小批次测试
        strategies_to_test=["cls", "mean", "max", "cls_mean"]  # 可以调整测试的策略
    )
    
    # 打印总结
    logger.info("\n" + "="*80)
    logger.info("特征提取总结")
    logger.info("="*80)
    
    for strategy_name, features in results.items():
        logger.info(f"\n策略: {strategy_name}")
        logger.info(f"  SMILES特征: {features['smiles_features'].shape}")
        logger.info(f"  蛋白质特征: {features['protein_features'].shape}")
        logger.info(f"  样本数量: {features['smiles_features'].shape[0]}")

if __name__ == "__main__":

    main()
