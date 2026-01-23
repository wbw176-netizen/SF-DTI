#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取脚本（变长存储 + float16版本）
使用本地预训练模型目录提取变长序列特征
"""

import os
import sys
import logging
import argparse
import numpy as np

# 确保可以导入项目根目录下的 two_stage_feature_extraction.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from two_stage_feature_extraction import TwoStageFeaturePipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(data_dir: str,
                     output_dir: str,
                     esm2_dir: str,
                     chemberta_dir: str,
                     prott5_dir: str,
                     batch_size: int = 4):
    """提取特征（变长存储 + float16）"""

    logger.info(f"\n{'=' * 60}")
    logger.info(f"开始特征提取: 变长序列特征 + float16")
    logger.info(f"{'=' * 60}")

    # 初始化管道（本地模型目录）
    pipeline = TwoStageFeaturePipeline(
        esm2_dir=esm2_dir,
        chemberta_dir=chemberta_dir,
        prott5_dir=prott5_dir,
    )

    # 提取特征
    all_features = pipeline.extract_features_for_all_splits(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    # 打印结果（变长特征列表）
    logger.info(f"特征提取完成:")
    for split, features in all_features.items():
        smiles_feat = features['smiles_features']
        esm2_feat = features['protein_features_esm2']
        prott5_feat = features['protein_features_prott5']
        
        # 计算统计信息
        smiles_lengths = [f.shape[0] for f in smiles_feat]
        esm2_lengths = [f.shape[0] for f in esm2_feat]
        prott5_lengths = [f.shape[0] for f in prott5_feat]
        
        logger.info(
            f"  {split}: "
            f"样本数={len(smiles_feat)}, "
            f"SMILES(平均长度={np.mean(smiles_lengths):.1f}, 最大={max(smiles_lengths)}), "
            f"ESM2(平均长度={np.mean(esm2_lengths):.1f}, 最大={max(esm2_lengths)}), "
            f"ProtT5(平均长度={np.mean(prott5_lengths):.1f}, 最大={max(prott5_lengths)}), "
            f"HDF5文件={features.get('h5_path', 'N/A')}"
        )

    return all_features


def extract_single_dataset(esm2_dir: str,
                           chemberta_dir: str,
                           prott5_dir: str,
                           data_dir: str,
                           output_dir: str,
                           batch_size: int = 4):
    """提取单个数据集的特征"""

    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return None

    try:
        features = extract_features(
            data_dir=data_dir,
            output_dir=output_dir,
            esm2_dir=esm2_dir,
            chemberta_dir=chemberta_dir,
            prott5_dir=prott5_dir,
            batch_size=batch_size,
        )
        return features
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        return None


def show_feature_statistics(features_dict: dict):
    """显示特征统计信息"""

    logger.info(f"\n{'=' * 60}")
    logger.info("特征统计信息")
    logger.info(f"{'=' * 60}")

    for split, split_features in features_dict.items():
        smiles_feat = split_features['smiles_features']
        esm2_feat = split_features['protein_features_esm2']
        prott5_feat = split_features['protein_features_prott5']
        
        # 计算序列长度统计
        smiles_lengths = [f.shape[0] for f in smiles_feat]
        esm2_lengths = [f.shape[0] for f in esm2_feat]
        prott5_lengths = [f.shape[0] for f in prott5_feat]
        
        def _aggregate_stats(feature_list):
            total_count = 0
            total_sum = 0.0
            total_sq_sum = 0.0

            for feature_array in feature_list:
                arr = np.asarray(feature_array, dtype=np.float32)
                if arr.size == 0:
                    continue

                total_count += arr.size
                total_sum += float(arr.sum(dtype=np.float64))
                total_sq_sum += float(np.square(arr, dtype=np.float64).sum(dtype=np.float64))

            if total_count == 0:
                return float("nan"), float("nan")

            mean = total_sum / total_count
            if total_count > 1:
                variance = (total_sq_sum - (total_sum ** 2) / total_count) / (total_count - 1)
                variance = max(variance, 0.0)
                std = variance ** 0.5
            else:
                std = 0.0

            return mean, std

        smiles_mean, smiles_std = _aggregate_stats(smiles_feat)
        esm2_mean, esm2_std = _aggregate_stats(esm2_feat)
        prott5_mean, prott5_std = _aggregate_stats(prott5_feat)
        
        logger.info(f"\n{split} 分割:")
        logger.info(f"  样本数: {len(smiles_feat)}")
        logger.info(f"  SMILES: 平均长度={np.mean(smiles_lengths):.1f}, 最大={max(smiles_lengths)}, "
                   f"特征均值={smiles_mean:.4f}, 标准差={smiles_std:.4f}")
        logger.info(f"  ESM2: 平均长度={np.mean(esm2_lengths):.1f}, 最大={max(esm2_lengths)}, "
                   f"特征均值={esm2_mean:.4f}, 标准差={esm2_std:.4f}")
        logger.info(f"  ProtT5: 平均长度={np.mean(prott5_lengths):.1f}, 最大={max(prott5_lengths)}, "
                   f"特征均值={prott5_mean:.4f}, 标准差={prott5_std:.4f}")
        if split_features.get('h5_path'):
            file_size_gb = os.path.getsize(split_features['h5_path']) / (1024 ** 3)
            logger.info(f"  HDF5文件大小: {file_size_gb:.2f} GB")


def batch_extract_random2(base_datasets_dir: str,
                          output_root_dir: str,
                          esm2_dir: str,
                          chemberta_dir: str,
                          prott5_dir: str,
                          batch_size: int = 4):
    """遍历 base_datasets_dir 子目录下 random2，批量提取特征（变长存储 + float16）。"""

    if not os.path.exists(base_datasets_dir):
        logger.error(f"数据根目录不存在: {base_datasets_dir}")
        return

    logger.info(f"开始批量提取: 根目录={base_datasets_dir}, 输出={output_root_dir}")
    logger.info(f"存储格式: 变长序列特征 + float16")

    for entry in sorted(os.listdir(base_datasets_dir)):
        dataset_dir = os.path.join(base_datasets_dir, entry)
        if not os.path.isdir(dataset_dir):
            continue

        random2_dir = os.path.join(dataset_dir, "random2")
        if not os.path.isdir(random2_dir):
            logger.info(f"跳过(无random2): {dataset_dir}")
            continue

        logger.info(f"处理数据集: {entry} -> {random2_dir}")
        output_dir = os.path.join(output_root_dir, entry)

        try:
            features = extract_features(
                data_dir=random2_dir,
                output_dir=output_dir,
                esm2_dir=esm2_dir,
                chemberta_dir=chemberta_dir,
                prott5_dir=prott5_dir,
                batch_size=batch_size,
            )
            for split, split_features in features.items():
                smiles_feat = split_features['smiles_features']
                esm2_feat = split_features['protein_features_esm2']
                prott5_feat = split_features['protein_features_prott5']
                logger.info(
                    f"  {entry}/{split}: "
                    f"样本数={len(smiles_feat)}, "
                    f"SMILES平均长度={np.mean([f.shape[0] for f in smiles_feat]):.1f}, "
                    f"ESM2平均长度={np.mean([f.shape[0] for f in esm2_feat]):.1f}, "
                    f"ProtT5平均长度={np.mean([f.shape[0] for f in prott5_feat]):.1f}"
                )
        except Exception as e:
            logger.error(f"数据集 {entry} 提取失败: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="特征提取脚本（变长存储 + float16版本）")
    parser.add_argument("mode", nargs="?", default="single",
                        choices=["single", "batch", "stats"],
                        help="运行模式: single=单个数据集, batch=批量处理, stats=显示统计")
    parser.add_argument("--data_dir", type=str, default="../datasets/Drugbank/random2",
                        help="包含 train/val/test.csv 的目录")
    parser.add_argument("--output_dir", type=str, default="../code/features/varlen",
                        help="输出目录")
    parser.add_argument("--esm2_dir", type=str, default="./esm2_3B_model", help="ESM2本地目录")
    parser.add_argument("--chemberta_dir", type=str, default="./chemberta_model", help="ChemBERTa本地目录")
    parser.add_argument("--prott5_dir", type=str, default="./protT5_model", help="ProtT5本地目录")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")

    args = parser.parse_args()

    if args.mode == "single":
        # 单个数据集提取
        features = extract_single_dataset(
            esm2_dir=args.esm2_dir,
            chemberta_dir=args.chemberta_dir,
            prott5_dir=args.prott5_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
        if features:
            show_feature_statistics(features)
    elif args.mode == "batch":
        # 批量提取
        batch_extract_random2(
            base_datasets_dir=args.data_dir,
            output_root_dir=args.output_dir,
            esm2_dir=args.esm2_dir,
            chemberta_dir=args.chemberta_dir,
            prott5_dir=args.prott5_dir,
            batch_size=args.batch_size,
        )
    elif args.mode == "stats":
        # 显示已提取特征的统计信息（需要从HDF5文件读取）
        logger.info("统计模式：需要从HDF5文件读取特征")
        logger.info("请使用 read_varlen_features.py 脚本查看特征统计")
    else:
        logger.info("未知模式")


