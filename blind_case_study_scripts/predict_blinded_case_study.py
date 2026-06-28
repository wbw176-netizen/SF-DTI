import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from models import SF_DTI
from utils import graph_collate_func


def parse_args():
    parser = argparse.ArgumentParser(description="Rank query-associated candidate pairs with a query-blinded SF-DTI model.")
    parser.add_argument("--data", required=True, help="Dataset name under ../datasets.")
    parser.add_argument("--split", default="random2")
    parser.add_argument("--precomputed_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics_output", default=None)
    parser.add_argument("--top_k", nargs="*", type=int, default=[1, 5, 10, 20, 50, 100])
    return parser.parse_args()


def load_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    return model


def compute_enrichment_metrics(ranked_df, top_ks):
    total_candidates = int(len(ranked_df))
    total_known = int(ranked_df["true_interaction"].sum())
    baseline_rate = total_known / total_candidates if total_candidates else 0.0

    rows = []
    for requested_k in top_ks:
        k = min(int(requested_k), total_candidates)
        if k <= 0:
            continue
        top_df = ranked_df.head(k)
        hits = int(top_df["true_interaction"].sum())
        hit_rate = hits / k
        recall = hits / total_known if total_known else 0.0
        enrichment_factor = hit_rate / baseline_rate if baseline_rate else np.nan
        rows.append(
            {
                "requested_k": int(requested_k),
                "evaluated_k": k,
                "candidate_count": total_candidates,
                "known_interactions": total_known,
                "baseline_known_rate": baseline_rate,
                "hits_at_k": hits,
                "hit_rate_at_k": hit_rate,
                "recall_at_k": recall,
                "enrichment_factor_at_k": enrichment_factor,
                "success_at_k": int(hits > 0),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    cfg = get_cfg_defaults()
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    data_dir = Path("..") / "datasets" / args.data / args.split
    test_csv = data_dir / "test.csv"
    test_ids_csv = data_dir / "test_ids.csv"
    test_feature_dir = Path(args.precomputed_dir) / "test"
    for input_path in (test_csv, test_ids_csv, test_feature_dir):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    df_test = pd.read_csv(test_csv)
    ids_df = pd.read_csv(test_ids_csv)
    if len(df_test) != len(ids_df):
        raise ValueError(f"test.csv and test_ids.csv length mismatch: {len(df_test)} vs {len(ids_df)}")

    dataset = DTIDataset(df_test.index.values, df_test, precomputed_features_dir=str(test_feature_dir))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=graph_collate_func,
    )

    checkpoint = args.checkpoint or str(Path("..") / "output" / "result" / args.data / args.split / "best_model.pth")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)

    model = SF_DTI(device=device, use_precomputed_features=True, **cfg).to(device)
    model = load_state(model, checkpoint, device)
    model.eval()

    probs = []
    labels = []
    with torch.no_grad():
        for v_d, v_p, y, drug_precomputed, protein_precomputed in loader:
            v_d = v_d.to(device)
            v_p = v_p.to(device)
            y = y.float().to(device)
            drug_precomputed = drug_precomputed.to(device) if drug_precomputed is not None else None
            if protein_precomputed is not None:
                protein_precomputed = tuple(item.to(device) for item in protein_precomputed)
            _, _, _, score = model(v_d, v_p, drug_precomputed, protein_precomputed)
            batch_probs = torch.sigmoid(score).detach().view(-1).cpu().numpy()
            probs.extend(batch_probs.tolist())
            labels.extend(y.detach().view(-1).cpu().numpy().astype(int).tolist())

    if len(probs) != len(ids_df):
        raise ValueError(f"Prediction count mismatch: predictions={len(probs)}, IDs={len(ids_df)}")
    if not np.array_equal(ids_df["Y"].to_numpy(dtype=int), np.asarray(labels, dtype=int)):
        raise ValueError("The test loader order does not match test_ids.csv.")

    out = ids_df[["drug_id", "protein_id", "Y"]].copy()
    out["predicted_score"] = np.array(probs, dtype=float)
    out["true_interaction"] = labels
    out = out.sort_values("predicted_score", ascending=False).reset_index(drop=True)

    output = args.output or str(Path("..") / "output" / "case_blind_predictions" / f"{args.data}_{args.split}.csv")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    metrics_df = compute_enrichment_metrics(out, args.top_k)
    metrics_output = args.metrics_output or str(output_path.with_name(output_path.stem + "_metrics.csv"))
    metrics_path = Path(metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    print(out.head(10).to_string(index=False))
    print(f"\nWrote ranked candidates: {output_path}")
    print("\nRetrieval metrics:")
    print(metrics_df.to_string(index=False))
    print(f"\nWrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
