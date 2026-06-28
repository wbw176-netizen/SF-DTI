# Query-Blinded DrugBank Case Studies

This directory contains the complete ranked candidate lists and retrieval
metrics used for the query-blinded case studies reported in the SF-DTI
manuscript.

## Reported Queries

| Query | Identifier | Setting | Ranked candidates | Metrics |
|---|---|---|---|---|
| NADH | DB00157 | Drug-centric | `nadh_case_study.csv` | `nadh_case_study_metrics.csv` |
| Beta-D-Glucose | DB02379 | Drug-centric | `glucose_case_study.csv` | `glucose_case_study_metrics.csv` |
| FAD | DB03147 | Drug-centric | `fad_case_study.csv` | `fad_case_study_metrics.csv` |
| H1R | P35367 | Target-centric | `h1r_case_study.csv` | `h1r_case_study_metrics.csv` |
| ADRA1C | P35348 | Target-centric | `adra1c_case_study.csv` | `adra1c_case_study_metrics.csv` |
| ADRA2A | P08913 | Target-centric | `adra2a_case_study.csv` | `adra2a_case_study_metrics.csv` |
| BACE1 | P56817 | Target-centric | `bace1_case_study.csv` | `bace1_case_study_metrics.csv` |
| HTR7 | P34969 | Target-centric | `htr7_case_study.csv` | `htr7_case_study_metrics.csv` |

ALOX5 is retained as an optional additional case and is not included in the
eight-query aggregate reported in the manuscript.

## Blinded Evaluation Protocol

DrugBank pairs are first deduplicated by DrugBank ID and UniProt ID. When
duplicate records contain conflicting labels, the positive annotation is
retained. For a drug-centric query, every pair containing the query drug is
removed before training. For a target-centric query, every pair containing the
query target is removed. The removed query-associated pairs form the candidate
set that is ranked by the trained model.

The model checkpoint is selected using the validation split. Candidate-set
labels are not used for model selection.

## Ranked-List Columns

- `drug_id`: DrugBank identifier.
- `protein_id`: UniProt identifier.
- `Y`: deduplicated DrugBank interaction annotation.
- `predicted_score`: SF-DTI interaction probability used for ranking.
- `true_interaction`: validation label used to calculate retrieval metrics.

Rows are sorted by `predicted_score` in descending order. `Y` and
`true_interaction` are retained separately to make the prediction and
evaluation stages explicit.

## Retrieval Metrics

- `hits_at_k`: number of confirmed interactions among the top-k candidates.
- `hit_rate_at_k`: `hits_at_k / evaluated_k`.
- `recall_at_k`: `hits_at_k / known_interactions`.
- `enrichment_factor_at_k`: top-k hit rate divided by the confirmed-interaction
  rate in the complete candidate set.
- `success_at_k`: 1 when at least one confirmed interaction is retrieved.

Across the eight reported queries, the ranked files reproduce Hit@1 of 8/8,
Hit@5 of 39/40, Hit@10 of 78/80, micro-averaged Recall@10 of 0.112, and mean
EF@10 of 1.063.

## Reproduction

The strict blinded workflow uses:

- `prepare_blinded_case_study_splits.py`
- `train_blinded_case_study.py`
- `predict_blinded_case_study.py`
- `run_blinded_case_studies.sh`

Prepare the deduplicated splits and aligned precomputed features:

```bash
cd code
python prepare_blinded_case_study_splits.py \
  --full_csv /path/to/DrugBank/full.csv \
  --raw_id_file /path/to/DrugBank/data.txt \
  --merged_feature_dir /path/to/DrugBank/Pretrained-features \
  --dataset_root ../datasets \
  --feature_root ../output/case_blind_features
```

The merged feature directory must contain:

```text
full_smiles_features.npy
full_protein_features_esm2.npy
full_protein_features_prott5.npy
```

Train and rank all eight manuscript queries:

```bash
bash run_blinded_case_studies.sh
```

The runner can be configured through environment variables, for example:

```bash
DEVICE=cuda:0 BATCH_SIZE=64 NUM_WORKERS=4 USE_AMP=1 \
  bash run_blinded_case_studies.sh
```

To run the optional ALOX5 case:

```bash
QUERY_LIST="alox5" bash run_blinded_case_studies.sh
```
