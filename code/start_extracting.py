#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging
import argparse
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from extractor import TwoStageFeaturePipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_features(data_dir: str,
                     output_dir: str,
                     esm2_dir: str,
                     chemberta_dir: str,
                     prott5_dir: str,
                     batch_size: int = 4):
    """Extract features with variable-length storage and float16."""

    logger.info(f"\n{'=' * 60}")
    logger.info("Start feature extraction: variable-length sequence features + float16")
    logger.info(f"{'=' * 60}")

    pipeline = TwoStageFeaturePipeline(
        esm2_dir=esm2_dir,
        chemberta_dir=chemberta_dir,
        prott5_dir=prott5_dir,
    )

    all_features = pipeline.extract_features_for_all_splits(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    logger.info("Feature extraction completed:")
    for split, features in all_features.items():
        smiles_feat = features['smiles_features']
        esm2_feat = features['protein_features_esm2']
        prott5_feat = features['protein_features_prott5']

        smiles_lengths = [f.shape[0] for f in smiles_feat]
        esm2_lengths = [f.shape[0] for f in esm2_feat]
        prott5_lengths = [f.shape[0] for f in prott5_feat]

        logger.info(
            f"  {split}: "
            f"num_samples={len(smiles_feat)}, "
            f"SMILES(avg_len={np.mean(smiles_lengths):.1f}, max={max(smiles_lengths)}), "
            f"ESM2(avg_len={np.mean(esm2_lengths):.1f}, max={max(esm2_lengths)}), "
            f"ProtT5(avg_len={np.mean(prott5_lengths):.1f}, max={max(prott5_lengths)}), "
            f"HDF5 file={features.get('h5_path', 'N/A')}"
        )

    return all_features


def extract_single_dataset(esm2_dir: str,
                           chemberta_dir: str,
                           prott5_dir: str,
                           data_dir: str,
                           output_dir: str,
                           batch_size: int = 4):
    """Extract features for a single dataset."""

    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
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
        logger.error(f"Feature extraction failed: {e}")
        return None


def show_feature_statistics(features_dict: dict):
    """Display feature statistics."""

    logger.info(f"\n{'=' * 60}")
    logger.info("Feature statistics")
    logger.info(f"{'=' * 60}")

    for split, split_features in features_dict.items():
        smiles_feat = split_features['smiles_features']
        esm2_feat = split_features['protein_features_esm2']
        prott5_feat = split_features['protein_features_prott5']

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

        logger.info(f"\n{split} split:")
        logger.info(
            f"  Number of samples: {len(smiles_feat)}"
        )
        logger.info(
            f"  SMILES: avg_len={np.mean(smiles_lengths):.1f}, max={max(smiles_lengths)}, "
            f"mean={smiles_mean:.4f}, std={smiles_std:.4f}"
        )
        logger.info(
            f"  ESM2: avg_len={np.mean(esm2_lengths):.1f}, max={max(esm2_lengths)}, "
            f"mean={esm2_mean:.4f}, std={esm2_std:.4f}"
        )
        logger.info(
            f"  ProtT5: avg_len={np.mean(prott5_lengths):.1f}, max={max(prott5_lengths)}, "
            f"mean={prott5_mean:.4f}, std={prott5_std:.4f}"
        )
        if split_features.get('h5_path'):
            file_size_gb = os.path.getsize(split_features['h5_path']) / (1024 ** 3)
            logger.info(f"  HDF5 file size: {file_size_gb:.2f} GB")


def batch_extract_random2(base_datasets_dir: str,
                          output_root_dir: str,
                          esm2_dir: str,
                          chemberta_dir: str,
                          prott5_dir: str,
                          batch_size: int = 4):
    """Batch extract features from random2 subdirectories under base_datasets_dir."""

    if not os.path.exists(base_datasets_dir):
        logger.error(f"Base dataset directory does not exist: {base_datasets_dir}")
        return

    logger.info(f"Start batch extraction: root={base_datasets_dir}, output={output_root_dir}")
    logger.info("Storage format: variable-length sequence features + float16")

    for entry in sorted(os.listdir(base_datasets_dir)):
        dataset_dir = os.path.join(base_datasets_dir, entry)
        if not os.path.isdir(dataset_dir):
            continue

        random2_dir = os.path.join(dataset_dir, "random2")
        if not os.path.isdir(random2_dir):
            logger.info(f"Skip (no random2): {dataset_dir}")
            continue

        logger.info(f"Processing dataset: {entry} -> {random2_dir}")
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
                    f"num_samples={len(smiles_feat)}, "
                    f"SMILES avg_len={np.mean([f.shape[0] for f in smiles_feat]):.1f}, "
                    f"ESM2 avg_len={np.mean([f.shape[0] for f in esm2_feat]):.1f}, "
                    f"ProtT5 avg_len={np.mean([f.shape[0] for f in prott5_feat]):.1f}"
                )
        except Exception as e:
            logger.error(f"Feature extraction failed for dataset {entry}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature extraction script (variable-length storage + float16 version)"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="single",
        choices=["single", "batch", "stats"],
        help="Run mode: single=single dataset, batch=batch processing, stats=show statistics"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../datasets/Drugbank/random2",
        help="Directory containing train/val/test.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../code/features/varlen",
        help="Output directory"
    )
    parser.add_argument(
        "--esm2_dir",
        type=str,
        default="./esm2_3B_model",
        help="Local directory for ESM2"
    )
    parser.add_argument(
        "--chemberta_dir",
        type=str,
        default="./chemberta_model",
        help="Local directory for ChemBERTa"
    )
    parser.add_argument(
        "--prott5_dir",
        type=str,
        default="./protT5_model",
        help="Local directory for ProtT5"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )

    args = parser.parse_args()

    if args.mode == "single":
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
        batch_extract_random2(
            base_datasets_dir=args.data_dir,
            output_root_dir=args.output_dir,
            esm2_dir=args.esm2_dir,
            chemberta_dir=args.chemberta_dir,
            prott5_dir=args.prott5_dir,
            batch_size=args.batch_size,
        )

    elif args.mode == "stats":
        logger.info("Statistics mode: features need to be loaded from HDF5 files")
        logger.info("Please use read_varlen_features.py to view feature statistics")

    else:
        logger.info("Unknown mode")
