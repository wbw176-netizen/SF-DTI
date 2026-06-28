import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd


QUERIES = {
    "nadh": {"setting": "drug", "id": "DB00157", "name": "NADH"},
    "glucose": {"setting": "drug", "id": "DB02379", "name": "Beta-D-Glucose"},
    "fad": {"setting": "drug", "id": "DB03147", "name": "FAD"},
    "h1r": {"setting": "target", "id": "P35367", "name": "H1R"},
    "adra1c": {"setting": "target", "id": "P35348", "name": "ADRA1C"},
    "adra2a": {"setting": "target", "id": "P08913", "name": "ADRA2A"},
    "alox5": {"setting": "target", "id": "P45059", "name": "ALOX5"},
    "bace1": {"setting": "target", "id": "P56817", "name": "BACE1"},
    "htr7": {"setting": "target", "id": "P34969", "name": "HTR7"},
}

REPORTED_QUERIES = (
    "nadh",
    "glucose",
    "fad",
    "h1r",
    "adra1c",
    "adra2a",
    "bace1",
    "htr7",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare query-blinded DrugBank case-study splits and aligned precomputed features."
    )
    parser.add_argument(
        "--full_csv",
        required=True,
        help="DrugBank full.csv with SMILES, Protein, Y columns.",
    )
    parser.add_argument(
        "--raw_id_file",
        required=True,
        help="Raw DrugBank file with drug_id protein_id SMILES Protein Y columns.",
    )
    parser.add_argument(
        "--merged_feature_dir",
        required=True,
        help="Directory containing full_smiles_features.npy, full_protein_features_esm2.npy, and full_protein_features_prott5.npy.",
    )
    parser.add_argument(
        "--dataset_root",
        default="../datasets",
        help="Output dataset root used by main.py.",
    )
    parser.add_argument(
        "--feature_root",
        default="../output/case_blind_features",
        help="Output feature root passed to --precomputed_dir.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--queries",
        nargs="*",
        default=list(REPORTED_QUERIES),
        choices=list(QUERIES.keys()),
        help="Queries to prepare. Defaults to the eight cases reported in the manuscript; ALOX5 is optional.",
    )
    return parser.parse_args()


def load_raw_id_table(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            parts = line.rstrip("\n").split()
            if len(parts) != 5:
                raise ValueError(f"Invalid raw row at line {line_no + 1}: expected 5 fields, got {len(parts)}")
            rows.append(
                {
                    "drug_id": parts[0],
                    "protein_id": parts[1],
                    "SMILES": parts[2],
                    "Protein": parts[3],
                    "Y": int(parts[4]),
                    "raw_index": line_no,
                }
            )
    return pd.DataFrame(rows)


def attach_ids_to_full(full_df, raw_df):
    raw_buckets = defaultdict(deque)
    for row in raw_df.itertuples(index=False):
        key = (row.SMILES, row.Protein, int(row.Y))
        raw_buckets[key].append((row.drug_id, row.protein_id, row.raw_index))

    drug_ids = []
    protein_ids = []
    raw_indices = []
    missing = 0
    for row in full_df.itertuples(index=False):
        key = (row.SMILES, row.Protein, int(row.Y))
        bucket = raw_buckets.get(key)
        if bucket:
            drug_id, protein_id, raw_index = bucket.popleft()
        else:
            drug_id, protein_id, raw_index = "", "", -1
            missing += 1
        drug_ids.append(drug_id)
        protein_ids.append(protein_id)
        raw_indices.append(raw_index)

    if missing:
        raise ValueError(f"Failed to map {missing} full.csv rows back to raw DrugBank IDs.")

    out = full_df.copy()
    out.insert(0, "full_index", np.arange(len(out), dtype=np.int64))
    out.insert(1, "drug_id", drug_ids)
    out.insert(2, "protein_id", protein_ids)
    out.insert(3, "raw_index", raw_indices)
    return out


def deduplicate_pairs(full_with_ids):
    # Keep a positive annotation when duplicate drug-target records disagree.
    ordered = full_with_ids.sort_values(["drug_id", "protein_id", "Y"], ascending=[True, True, False])
    dedup = ordered.drop_duplicates(subset=["drug_id", "protein_id"], keep="first")
    return dedup.sort_values("full_index").reset_index(drop=True)


def stratified_val_indices(df, val_fraction, seed):
    rng = np.random.default_rng(seed)
    val_indices = []
    for label in sorted(df["Y"].unique()):
        label_indices = df.index[df["Y"] == label].to_numpy()
        rng.shuffle(label_indices)
        n_val = max(1, int(round(len(label_indices) * val_fraction)))
        val_indices.extend(label_indices[:n_val].tolist())
    return set(val_indices)


def save_split(df, path, prefix, arrays):
    path.mkdir(parents=True, exist_ok=True)
    csv_df = df[["SMILES", "Protein", "Y"]].reset_index(drop=True)
    csv_df.to_csv(path / f"{prefix}.csv", index=False)

    feature_dir = path
    full_indices = df["full_index"].to_numpy(dtype=np.int64)
    np.save(feature_dir / f"{prefix}_smiles_features.npy", arrays["smiles"][full_indices])
    np.save(feature_dir / f"{prefix}_protein_features_esm2.npy", arrays["esm2"][full_indices])
    np.save(feature_dir / f"{prefix}_protein_features_prott5.npy", arrays["prott5"][full_indices])
    np.save(feature_dir / f"{prefix}_labels.npy", df["Y"].to_numpy(dtype=np.int64))


def main():
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val_fraction must be between 0 and 1.")

    full_csv = Path(args.full_csv)
    raw_id_file = Path(args.raw_id_file)
    feature_dir = Path(args.merged_feature_dir)
    dataset_root = Path(args.dataset_root)
    feature_root = Path(args.feature_root)

    for input_path in (full_csv, raw_id_file, feature_dir):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    full_df = pd.read_csv(full_csv)
    if list(full_df.columns) != ["SMILES", "Protein", "Y"]:
        raise ValueError(f"Unexpected full.csv columns: {list(full_df.columns)}")

    raw_df = load_raw_id_table(raw_id_file)
    if len(raw_df) != len(full_df):
        raise ValueError(f"Row count mismatch: full.csv={len(full_df)}, raw_id_file={len(raw_df)}")

    full_with_ids = attach_ids_to_full(full_df, raw_df)
    dedup = deduplicate_pairs(full_with_ids)

    arrays = {
        "smiles": np.load(feature_dir / "full_smiles_features.npy", mmap_mode="r"),
        "esm2": np.load(feature_dir / "full_protein_features_esm2.npy", mmap_mode="r"),
        "prott5": np.load(feature_dir / "full_protein_features_prott5.npy", mmap_mode="r"),
    }
    for feature_name, feature_array in arrays.items():
        if len(feature_array) != len(full_df):
            raise ValueError(
                f"{feature_name} feature count mismatch: features={len(feature_array)}, full.csv={len(full_df)}"
            )

    summaries = []
    for query_key in args.queries:
        query = QUERIES[query_key]
        data_name = f"Drugbank_case_blind_{query_key}"
        data_dir = dataset_root / data_name / "random2"
        feat_dir = feature_root / data_name / "random2"

        if query["setting"] == "drug":
            candidate_mask = dedup["drug_id"] == query["id"]
        else:
            candidate_mask = dedup["protein_id"] == query["id"]

        candidate_df = dedup[candidate_mask].copy()
        if candidate_df.empty:
            raise ValueError(f"No candidate pairs found for {query_key} ({query['id']}).")

        pool_df = dedup[~candidate_mask].copy().reset_index(drop=True)
        val_idx = stratified_val_indices(pool_df, args.val_fraction, args.seed)
        val_df = pool_df.loc[sorted(val_idx)].copy()
        train_df = pool_df.drop(index=sorted(val_idx)).copy()

        (data_dir).mkdir(parents=True, exist_ok=True)
        train_df[["SMILES", "Protein", "Y"]].to_csv(data_dir / "train.csv", index=False)
        val_df[["SMILES", "Protein", "Y"]].to_csv(data_dir / "val.csv", index=False)
        candidate_df[["SMILES", "Protein", "Y"]].to_csv(data_dir / "test.csv", index=False)
        candidate_df[
            ["drug_id", "protein_id", "SMILES", "Protein", "Y", "full_index", "raw_index"]
        ].to_csv(data_dir / "test_ids.csv", index=False)

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", candidate_df)]:
            save_split(split_df, feat_dir / split_name, split_name, arrays)

        summary = {
            "query_key": query_key,
            "query_name": query["name"],
            "query_id": query["id"],
            "setting": query["setting"],
            "data_name": data_name,
            "train_pairs": int(len(train_df)),
            "val_pairs": int(len(val_df)),
            "candidate_pairs_removed_from_training": int(len(candidate_df)),
            "removed_confirmed_pairs": int(candidate_df["Y"].sum()),
            "non_confirmed_candidates": int((candidate_df["Y"] == 0).sum()),
            "dataset_dir": str(data_dir),
            "feature_dir": str(feat_dir),
        }
        summaries.append(summary)
        with open(data_dir / "split_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    summary_df = pd.DataFrame(summaries)
    summary_path = dataset_root / "Drugbank_case_blind_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
