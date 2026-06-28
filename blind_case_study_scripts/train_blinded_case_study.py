import argparse
import os
import warnings
from datetime import datetime
from time import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import SF_DTI
from trainer import Trainer
from utils import graph_collate_func, mkdir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train SF-DTI on a query-blinded case-study split.")
    parser.add_argument("--data", required=True, help="Dataset name under ../datasets, e.g. Drugbank_case_blind_nadh")
    parser.add_argument("--split", default="random2")
    parser.add_argument("--precomputed_dir", required=True, help="Aligned precomputed feature directory.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    torch.cuda.empty_cache()

    cfg = get_cfg_defaults()
    cfg.SOLVER.SEED = args.seed
    cfg.SOLVER.BATCH_SIZE = args.batch_size
    cfg.SOLVER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.PATIENCE = args.patience
    if args.epochs is not None:
        cfg.SOLVER.MAX_EPOCH = args.epochs

    set_seed(cfg.SOLVER.SEED)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    data_folder = os.path.join("..", "datasets", args.data, args.split)
    train_path = os.path.join(data_folder, "train.csv")
    val_path = os.path.join(data_folder, "val.csv")
    test_path = os.path.join(data_folder, "test.csv")

    train_precomputed_dir = os.path.join(args.precomputed_dir, "train")
    val_precomputed_dir = os.path.join(args.precomputed_dir, "val")
    test_precomputed_dir = os.path.join(args.precomputed_dir, "test")

    required_paths = [
        train_path,
        val_path,
        test_path,
        train_precomputed_dir,
        val_precomputed_dir,
        test_precomputed_dir,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError("Missing required case-study inputs:\n" + "\n".join(missing_paths))

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    output_dir = cfg.RESULT.OUTPUT_DIR + f"{args.data}/{args.split}"
    mkdir(output_dir)

    print("=" * 80)
    print("SF-DTI query-blinded case-study training")
    print(f"data={args.data}, split={args.split}")
    print(f"train/val/test={len(df_train)}/{len(df_val)}/{len(df_test)}")
    print(f"precomputed_dir={args.precomputed_dir}")
    print(f"batch_size={cfg.SOLVER.BATCH_SIZE}, num_workers={cfg.SOLVER.NUM_WORKERS}")
    print(f"epochs={cfg.SOLVER.MAX_EPOCH}, patience={cfg.SOLVER.PATIENCE}, seed={cfg.SOLVER.SEED}")
    print(f"device={device}, amp={args.amp}")
    print(f"output={output_dir}")
    print("=" * 80)

    train_dataset = DTIDataset(df_train.index.values, df_train, precomputed_features_dir=train_precomputed_dir)
    val_dataset = DTIDataset(df_val.index.values, df_val, precomputed_features_dir=val_precomputed_dir)
    test_dataset = DTIDataset(df_test.index.values, df_test, precomputed_features_dir=test_precomputed_dir)

    train_params = {
        "batch_size": cfg.SOLVER.BATCH_SIZE,
        "shuffle": True,
        "num_workers": cfg.SOLVER.NUM_WORKERS,
        "drop_last": True,
        "collate_fn": graph_collate_func,
    }
    eval_params = dict(train_params)
    eval_params["shuffle"] = False
    eval_params["drop_last"] = False

    train_loader = DataLoader(train_dataset, **train_params)
    val_loader = DataLoader(val_dataset, **eval_params)
    test_loader = DataLoader(test_dataset, **eval_params)

    model = SF_DTI(device=device, use_precomputed_features=True, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = False

    trainer = Trainer(
        model,
        opt,
        device,
        train_loader,
        val_loader,
        test_loader,
        args.data,
        args.split,
        use_amp=args.amp,
        **cfg,
    )
    result = trainer.train()

    with open(os.path.join(output_dir, "model_architecture.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(model))
    with open(os.path.join(output_dir, "config.txt"), "w", encoding="utf-8") as handle:
        handle.write(str(dict(cfg)))

    return result


if __name__ == "__main__":
    print(f"start time: {datetime.now()}")
    start = time()
    main()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(time() - start, 2)}s")
