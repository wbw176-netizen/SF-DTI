import argparse
import json
import math
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from cdan_ablation import (
    ClusterPrecomputedDTIDataset,
    best_f1_threshold,
    cfg_to_plain,
    compute_metrics,
    evaluate,
    find_split_dir,
    graph_collate_func,
    infer_precomputed_root,
    move_batch,
    normalize_path,
    parse_list,
    resolve_precomputed_root,
    set_seed,
)
from configs import get_cfg_defaults
from models import SF_DTI


try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


class RandomMultilinearMap(nn.Module):
    def __init__(self, feature_dim, prediction_dim=2, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("feature_matrix", torch.randn(feature_dim, output_dim))
        self.register_buffer("prediction_matrix", torch.randn(prediction_dim, output_dim))

    def forward(self, features, predictions):
        feature_projection = features @ self.feature_matrix
        prediction_projection = predictions @ self.prediction_matrix
        return (feature_projection * prediction_projection) / math.sqrt(float(self.output_dim))


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class CDANAdversary(nn.Module):
    def __init__(self, feature_dim, random_dim=256, hidden_dim=256, dropout=0.1, use_entropy=True, feature_norm=False):
        super().__init__()
        self.use_entropy = use_entropy
        self.feature_norm = feature_norm
        self.conditioning = RandomMultilinearMap(feature_dim, prediction_dim=2, output_dim=random_dim)
        self.discriminator = DomainDiscriminator(random_dim, hidden_dim=hidden_dim, dropout=dropout)

    @staticmethod
    def prediction_probabilities(logits):
        positive = torch.sigmoid(torch.clamp(logits.view(-1, 1), min=-30.0, max=30.0))
        positive = torch.clamp(positive, min=1e-6, max=1.0 - 1e-6)
        return torch.cat([1.0 - positive, positive], dim=1)

    def forward(self, features, logits, grl_weight):
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        if self.feature_norm:
            features = F.normalize(features, p=2, dim=1, eps=1e-8)
        probabilities = self.prediction_probabilities(logits)
        domain_features = self.conditioning(features, probabilities)
        domain_features = GradientReverse.apply(domain_features, grl_weight)
        return self.discriminator(domain_features), probabilities


def model_features_logits(model, batch):
    graph, protein, _, drug_precomputed, protein_precomputed = batch
    _, _, features, logits = model(
        graph,
        protein,
        drug_precomputed=drug_precomputed,
        protein_precomputed=protein_precomputed,
        mode="extract_features",
    )
    return features, logits.view(-1)


def cdan_domain_loss(adversary, source_features, source_logits, target_features, target_logits, grl_weight):
    features = torch.cat([source_features, target_features], dim=0)
    logits = torch.cat([source_logits, target_logits], dim=0)
    domain_logits, probabilities = adversary(features, logits, grl_weight)
    domain_labels = torch.cat(
        [
            torch.zeros(source_features.size(0), dtype=torch.long, device=features.device),
            torch.ones(target_features.size(0), dtype=torch.long, device=features.device),
        ],
        dim=0,
    )
    losses = F.cross_entropy(domain_logits, domain_labels, reduction="none")
    if adversary.use_entropy:
        entropy = -torch.sum(probabilities.detach() * torch.log(probabilities.detach() + 1e-8), dim=1)
        weights = 1.0 + torch.exp(-entropy)
        losses = losses * (weights / weights.mean().clamp_min(1e-8))
    return losses.mean()


def schedule_weight(epoch, step, steps_per_epoch, max_epochs, init_epoch, mode):
    if epoch < init_epoch:
        return 0.0
    if mode == "drugban":
        return 1.0
    total_steps = max(1, (max_epochs - init_epoch + 1) * steps_per_epoch)
    current = max(0, (epoch - init_epoch) * steps_per_epoch + step)
    progress = min(1.0, current / total_steps)
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def finalize_paths(args):
    mlmdti_root = normalize_path(args.mlmdti_root)
    args.data_dir = str(normalize_path(args.data_dir, mlmdti_root) if args.data_dir else mlmdti_root / "datasets")
    args.output_root = str(normalize_path(args.output_root, mlmdti_root) if args.output_root else mlmdti_root / "output")
    if args.precomputed_root is None:
        args.precomputed_root = str(infer_precomputed_root(args.output_root, args.data))
    else:
        args.precomputed_root = str(normalize_path(args.precomputed_root, mlmdti_root))
    args.output_dir = str(
        normalize_path(args.output_dir, mlmdti_root) if args.output_dir else Path(args.output_root) / "cdan_cross_domain"
    )
    return args


def build_loaders(args, config, seed):
    data_dir = Path(args.data_dir) / args.data / "cluster"
    feature_root = resolve_precomputed_root(args.precomputed_root)
    args.precomputed_root = str(feature_root)
    train_feat_dir, train_feat_alias = find_split_dir(feature_root, "train")
    val_feat_dir, val_feat_alias = find_split_dir(feature_root, "val")
    test_feat_dir, test_feat_alias = find_split_dir(feature_root, "test")

    paths = {
        "source_train_csv": data_dir / "source_train.csv",
        "target_train_csv": data_dir / "target_train.csv",
        "target_test_csv": data_dir / "target_test.csv",
        "train_feat": train_feat_dir,
        "val_feat": val_feat_dir,
        "test_feat": test_feat_dir,
    }
    for name, path in paths.items():
        if path is None or not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    source_df = pd.read_csv(paths["source_train_csv"])
    source_indices = np.arange(len(source_df))
    if args.val_mode == "source_holdout":
        train_idx, val_idx = train_test_split(
            source_indices,
            test_size=args.val_fraction,
            random_state=seed,
            stratify=source_df["Y"] if "Y" in source_df.columns else None,
        )
    else:
        train_idx, val_idx = source_indices, None

    source_train_ds = ClusterPrecomputedDTIDataset(
        paths["source_train_csv"],
        paths["train_feat"],
        ("source_train", "train"),
        indices=train_idx,
        max_drug_nodes=config["DRUG"]["MAX_NODES"],
    )
    target_train_ds = ClusterPrecomputedDTIDataset(
        paths["target_train_csv"],
        paths["val_feat"],
        ("target_train", "val"),
        max_drug_nodes=config["DRUG"]["MAX_NODES"],
    )
    test_ds = ClusterPrecomputedDTIDataset(
        paths["target_test_csv"],
        paths["test_feat"],
        ("target_test", "test"),
        max_drug_nodes=config["DRUG"]["MAX_NODES"],
    )
    if args.val_mode == "source_holdout":
        val_ds = ClusterPrecomputedDTIDataset(
            paths["source_train_csv"],
            paths["train_feat"],
            ("source_train", "train"),
            indices=val_idx,
            max_drug_nodes=config["DRUG"]["MAX_NODES"],
        )
    elif args.val_mode == "target_train":
        val_ds = target_train_ds
    elif args.val_mode == "target_test":
        val_ds = test_ds
    else:
        raise ValueError(f"Unsupported val_mode: {args.val_mode}")

    common = {
        "batch_size": args.batch_size or config["SOLVER"]["BATCH_SIZE"],
        "num_workers": args.num_workers if args.num_workers is not None else config["SOLVER"]["NUM_WORKERS"],
        "collate_fn": graph_collate_func,
    }
    source_loader = DataLoader(source_train_ds, shuffle=True, drop_last=True, **common)
    target_loader = DataLoader(target_train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)
    target_train_role = "unlabeled target DA"
    target_test_role = "final test"
    if args.val_mode == "target_train":
        target_train_role += " + validation"
    elif args.val_mode == "target_test":
        target_test_role = "validation + final test"
    print(
        "Split mapping: "
        f"source_train.csv -> supervised source train ({train_feat_alias}), "
        f"target_train.csv -> {target_train_role} ({val_feat_alias}), "
        f"target_test.csv -> {target_test_role} ({test_feat_alias})"
    )
    return source_loader, target_loader, val_loader, test_loader


def train_one_epoch(model, adversary, source_loader, target_loader, optimizer, scaler, device, args, epoch, max_epochs):
    model.train()
    adversary.train()
    use_amp = args.amp and device.type == "cuda"
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)
    steps_per_epoch = max(len(source_loader), len(target_loader))
    total_loss, total_task_loss, total_da_loss, total_da_lambda = 0.0, 0.0, 0.0, 0.0

    for step in range(1, steps_per_epoch + 1):
        source_batch = next(source_iter)
        source_batch = move_batch(source_batch, device)
        target_batch = move_batch(next(target_iter), device)
        _, _, source_labels, _, _ = source_batch
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            source_features, source_logits = model_features_logits(model, source_batch)
            task_loss = F.binary_cross_entropy_with_logits(source_logits, source_labels)
            target_features, target_logits = model_features_logits(model, target_batch)
            schedule = schedule_weight(epoch, step, steps_per_epoch, max_epochs, args.da_init_epoch, args.grl_mode)
            da_lambda = args.da_lambda_scale * schedule if args.da_loss_schedule else 1.0
            grl_weight = da_lambda if args.da_loss_schedule else schedule
            da_loss = cdan_domain_loss(
                adversary,
                source_features,
                source_logits,
                target_features,
                target_logits,
                grl_weight=grl_weight,
            )
            if not torch.isfinite(da_loss):
                da_loss = task_loss.new_zeros(())
            loss = task_loss + args.da_weight * da_lambda * da_loss

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_da_loss += da_loss.item()
        total_da_lambda += float(da_lambda)

    denom = max(1, steps_per_epoch)
    return {
        "train_loss": total_loss / denom,
        "task_loss": total_task_loss / denom,
        "domain_loss": total_da_loss / denom,
        "da_lambda": total_da_lambda / denom,
    }


def is_better(metric, best, min_delta):
    if math.isnan(metric):
        return False
    return best is None or metric > best + min_delta


def run_one(args, seed):
    cfg = get_cfg_defaults()
    config = cfg_to_plain(cfg)
    config["SOLVER"]["SEED"] = seed
    if args.epochs is not None:
        config["SOLVER"]["MAX_EPOCH"] = args.epochs
    if args.batch_size is not None:
        config["SOLVER"]["BATCH_SIZE"] = args.batch_size
    if args.num_workers is not None:
        config["SOLVER"]["NUM_WORKERS"] = args.num_workers
    if args.lr is not None:
        config["SOLVER"]["LR"] = args.lr
    if args.weight_decay is not None:
        config["SOLVER"]["WEIGHT_DECAY"] = args.weight_decay

    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    use_amp = args.amp and device.type == "cuda"
    run_dir = Path(args.output_dir) / args.data / "cluster" / f"cdan_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        print(f"[SKIP] existing result: {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    run_dir.mkdir(parents=True, exist_ok=True)
    source_loader, target_loader, val_loader, test_loader = build_loaders(args, config, seed)

    model = SF_DTI(device=device, use_precomputed_features=True, **config).to(device)
    adversary = CDANAdversary(
        feature_dim=config["DECODER"]["IN_DIM"],
        random_dim=args.random_dim,
        hidden_dim=args.domain_hidden_dim,
        dropout=args.domain_dropout,
        use_entropy=args.use_entropy and not args.no_entropy,
        feature_norm=args.feature_norm,
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": config["SOLVER"]["LR"]},
            {"params": adversary.parameters(), "lr": args.da_lr},
        ],
        weight_decay=config["SOLVER"]["WEIGHT_DECAY"],
    )
    scaler = GradScaler(enabled=use_amp)

    max_epochs = config["SOLVER"]["MAX_EPOCH"]
    patience = args.patience if args.patience is not None else config["SOLVER"].get("PATIENCE", 8)
    min_delta = args.min_delta if args.min_delta is not None else config["SOLVER"].get("MIN_DELTA", 0.001)
    best_val, best_epoch, stale = None, 0, 0
    best_path = run_dir / "best_model.pt"
    history = []

    print("=" * 80)
    print("CDAN cross-domain experiment")
    print(f"Dataset: {args.data}/cluster")
    print(f"Train: source_train supervised + target_train unlabeled domain adaptation")
    print(f"Val mode: {args.val_mode}")
    print(f"Test: target_test")
    print(f"Early stopping: {args.early_stop}")
    print(f"Precomputed root: {args.precomputed_root}")
    print(
        f"DA weight: {args.da_weight}, DA init epoch: {args.da_init_epoch}, "
        f"GRL mode={args.grl_mode}, entropy={args.use_entropy and not args.no_entropy}, "
        f"feature_norm={args.feature_norm}, da_loss_schedule={args.da_loss_schedule}, "
        f"da_lambda_scale={args.da_lambda_scale}"
    )
    print(f"Seed: {seed}, device: {device}, AMP={use_amp}")
    print(f"Output: {run_dir}")
    print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        if config["SOLVER"].get("USE_LD", True) and epoch > 1 and epoch % config["SOLVER"]["DECAY_INTERVAL"] == 0:
            for group in optimizer.param_groups:
                group["lr"] *= config["SOLVER"]["LR_DECAY"]

        train_metrics = train_one_epoch(
            model, adversary, source_loader, target_loader, optimizer, scaler, device, args, epoch, max_epochs
        )
        val_eval = evaluate(model, val_loader, device, use_amp)
        val_threshold = best_f1_threshold(val_eval["labels"], val_eval["probs"])
        val_metrics = compute_metrics(val_eval["labels"], val_eval["probs"], val_threshold)
        val_metrics["loss"] = val_eval["loss"]
        history.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(
            f"Epoch {epoch:03d} train={train_metrics['train_loss']:.4f} "
            f"task={train_metrics['task_loss']:.4f} domain={train_metrics['domain_loss']:.4f} "
            f"da_lambda={train_metrics['da_lambda']:.4f} "
            f"val_AUROC={val_metrics['AUROC']:.4f} val_AUPRC={val_metrics['AUPRC']:.4f}"
        )

        if is_better(val_metrics["AUROC"], best_val, min_delta):
            best_val = val_metrics["AUROC"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "adversary_state_dict": adversary.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                    "args": vars(args),
                    "seed": seed,
                },
                best_path,
            )
        else:
            stale += 1
            if args.early_stop and stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    adversary.load_state_dict(checkpoint["adversary_state_dict"])

    val_eval = evaluate(model, val_loader, device, use_amp)
    test_eval = evaluate(model, test_loader, device, use_amp)
    threshold = best_f1_threshold(val_eval["labels"], val_eval["probs"])
    val_metrics = compute_metrics(val_eval["labels"], val_eval["probs"], threshold)
    test_metrics = compute_metrics(test_eval["labels"], test_eval["probs"], threshold)

    threshold_mode = "drugban_target_test_f1" if args.val_mode == "target_test" else "val_f1"
    result = {
        "dataset": args.data,
        "split": "cluster",
        "method": "cdan",
        "seed": seed,
        "best_epoch": best_epoch,
        "train_split": "source_train",
        "target_da_split": "target_train",
        "val_mode": args.val_mode,
        "test_split": "target_test",
        "threshold_mode": threshold_mode,
        "val_AUROC": val_metrics["AUROC"],
        "val_AUPRC": val_metrics["AUPRC"],
        "val_loss": val_eval["loss"],
        "AUROC": test_metrics["AUROC"],
        "AUPRC": test_metrics["AUPRC"],
        "F1": test_metrics["F1"],
        "Precision": test_metrics["Precision"],
        "Recall": test_metrics["Recall"],
        "MCC": test_metrics["MCC"],
        "Accuracy": test_metrics["Accuracy"],
        "Sensitivity": test_metrics["Sensitivity"],
        "Specificity": test_metrics["Specificity"],
        "Threshold": threshold,
        "test_loss": test_eval["loss"],
        "da_weight": args.da_weight,
        "da_init_epoch": args.da_init_epoch,
        "grl_mode": args.grl_mode,
        "da_loss_schedule": args.da_loss_schedule,
        "da_lambda_scale": args.da_lambda_scale,
        "use_entropy": args.use_entropy and not args.no_entropy,
        "feature_norm": args.feature_norm,
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"config": config, "args": vars(args)}, handle, indent=2)
    pd.DataFrame({"label": test_eval["labels"], "prob": test_eval["probs"]}).to_csv(
        run_dir / "target_test_predictions.csv", index=False
    )

    print(
        f"[RESULT] seed={seed} AUROC={result['AUROC']:.4f} AUPRC={result['AUPRC']:.4f} "
        f"F1={result['F1']:.4f} MCC={result['MCC']:.4f}"
    )
    return result


def summarize(results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(results)
    runs_path = output_dir / "summary_cdan_runs.csv"
    runs.to_csv(runs_path, index=False)
    metric_cols = ["AUROC", "AUPRC", "F1", "MCC", "Accuracy", "Sensitivity", "Specificity"]
    summary = runs.groupby(["dataset", "method"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary_path = output_dir / "summary_cdan_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved run summary: {runs_path}")
    print(f"Saved mean/std summary: {summary_path}")
    print(summary)


def parse_args():
    cfg = get_cfg_defaults()
    parser = argparse.ArgumentParser(description="Run clean CDAN cross-domain experiment for SF-DTI cluster splits")
    parser.add_argument("--data", required=True, help="Dataset name under data_dir, e.g. biosnap or bindingdb")
    parser.add_argument("--mlmdti_root", default="/root/autodl-tmp/MLMDTI")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--precomputed_root", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seeds", default=str(cfg.SOLVER.SEED))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--min_delta", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--val_mode", default="target_test", choices=["target_test", "target_train", "source_holdout"])
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--da_weight", type=float, default=1.0)
    parser.add_argument("--da_lr", type=float, default=5e-5)
    parser.add_argument("--da_init_epoch", type=int, default=10)
    parser.add_argument("--grl_mode", default="drugban", choices=["drugban", "progressive"])
    parser.add_argument("--random_dim", type=int, default=256)
    parser.add_argument("--domain_hidden_dim", type=int, default=256)
    parser.add_argument("--domain_dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--feature_norm", action="store_true")
    parser.add_argument("--da_loss_schedule", action="store_true")
    parser.add_argument("--da_lambda_scale", type=float, default=1.0)
    parser.add_argument("--use_entropy", action="store_true")
    parser.add_argument("--no_entropy", action="store_true")
    return parser.parse_args()


def main():
    args = finalize_paths(parse_args())
    results = [run_one(args, int(seed)) for seed in parse_list(args.seeds)]
    summarize(results, Path(args.output_dir) / args.data / "cluster")


if __name__ == "__main__":
    main()
