# SF-DTI

SF-DTI is a drug-target interaction (DTI) prediction framework that integrates
multi-encoder protein representations, structure-semantic drug representation
alignment, and cascaded cross-modal signal propagation for interaction
prediction.

This repository provides the code, dataset splits, processed resources, and
result files used in the revised manuscript.

## Repository Access

```text
https://github.com/wbw176-netizen/SF-DTI
```

## Repository Structure

```text
SF-DTI/
├── code/                         # Model, training, evaluation, and utility scripts
├── datasets/                     # Dataset splits used in the experiments
├── output/                       # Pre-extracted features and saved result files
├── visualization/                # UMAP and case-study visualization resources
├── case_study_scores/            # Ranked prediction scores for case studies
├── PDB_FILES/                    # PDB-related files for structural interpretability
└── README.md
```

The exact folder names may differ slightly between the anonymous review package
and the final public release, but the same code, datasets, and result resources
are provided.

## Environment

The experiments were conducted with Python and PyTorch on NVIDIA GPUs. A CUDA
GPU is recommended because pretrained feature extraction and model training are
computationally expensive.

Core dependencies include:

```text
python >= 3.9
pytorch
dgl
dgllife
rdkit
transformers
huggingface_hub
numpy
pandas
scikit-learn
scipy
h5py
yacs
tqdm
prettytable
matplotlib
seaborn
umap-learn
```

Example installation:

```bash
conda create -n sfdti python=3.9
conda activate sfdti

# Install PyTorch according to your CUDA version from:
# https://pytorch.org/get-started/locally/

pip install dgl dgllife rdkit-pypi transformers huggingface_hub
pip install numpy pandas scikit-learn scipy h5py yacs tqdm prettytable
pip install matplotlib seaborn umap-learn
```

If your platform uses a different DGL/PyTorch/CUDA combination, install the
compatible versions following the official DGL and PyTorch instructions.

## Datasets

The repository contains the processed splits used in the manuscript. Each split
is stored as CSV files under `datasets/{dataset}/{split}/`.

Typical files are:

```text
train.csv
val.csv
test.csv
full.csv
```

Each CSV contains the following main columns:

```text
SMILES, Protein, Y
```

where `SMILES` is the molecular SMILES string, `Protein` is the amino-acid
sequence, and `Y` is the binary interaction label. Some split files also contain
additional columns such as drug or target cluster identifiers.

## Dataset Availability
The full datasets used in this study (including training/testing splits and raw files) are hosted on Google Drive due to file size limitations. 
You can download them from the following link:
[**Download Full Datasets (Google Drive)**](https://drive.google.com/drive/folders/1IpJ8g2GJPoX70LL9fgDRYGFlmS3e6Cob?usp=drive_link)
Please download and unzip the data into the `datasets/` directory before running the code.


The main datasets used in the paper include:

```text
BindingDB
BioSNAP
DrugBank
Human
KIBA
C.elegans
```

## Pretrained Encoders

SF-DTI uses the following pretrained encoders:

| Modality | Encoder | Source |
| --- | --- | --- |
| Drug SMILES | ChemBERTa | `DeepChem/ChemBERTa-77M-MLM` |
| Protein sequence | ESM-2 | `facebook/esm2_t36_3B_UR50D` |
| Protein sequence | ProtT5 | `Rostlab/prot_t5_xl_uniref50` |

Download the model weights from Hugging Face and place them in local
directories, for example:

```text
models/
├── chemberta_model/
├── esm2_model/
└── protT5_model/
```

The script `code/download_models.py` can be used to download Hugging Face model
snapshots when internet access is available. On offline servers, copy the
downloaded model directories to the expected paths.

## Pre-extracted Features

The main experiments use pre-extracted ChemBERTa, ESM-2, and ProtT5 features to
avoid repeatedly running large pretrained encoders during every training run.
The expected feature directory used by the training scripts is:

```text
output/pooling_strategies_offline3/{dataset}/mean_mean/
```

For the clustering-based cross-domain setting, the expected feature directory is:

```text
output/cdan_cluster_features/{dataset}_cluster/
```

If pre-extracted features are provided in the review package, reviewers can run
the model directly without re-extracting PLM features.

## Running SF-DTI

All commands below assume that the current directory is `code/`.

```bash
cd code
```

### Random Split Evaluation

Run SF-DTI on one dataset under the random split setting:

```bash
python main.py \
  --data bindingdb \
  --split random2 \
  --use_precomputed \
  --precomputed_dir ../output/pooling_strategies_offline3/bindingdb/mean_mean \
  --amp
```

To run the repeated random-split experiments for the benchmark datasets:

```bash
bash run_random_split_extra_seeds.sh
```

Results are saved under:

```text
output/random_split_result/
```

### Five-fold Cross-validation

Run the five-fold cross-validation experiments:

```bash
bash run_five_fold_cv_all.sh
```

Results are saved under:

```text
output/five_fold_cv_result/
```

### Unseen-drug and Unseen-target Evaluation

Run the cold-start experiments:

```bash
bash run_coldstart_unseen_all.sh
```

Results are saved under:

```text
output/coldstart_unseen_result/
```

### Clustering-based Cross-domain Evaluation

First prepare the cluster-split features if they are not already provided:

```bash
bash run_extract_cdan_cluster_features.sh
```

Then run the clustering-based cross-domain experiment:

```bash
DATASET=bindingdb bash run_cdan_cross_domain_amp.sh
DATASET=biosnap bash run_cdan_cross_domain_amp.sh
```

Results are saved under the output directory specified in the script or by the
`OUTPUT_DIR` environment variable.

## Case Studies

The case-study prediction scores are provided in:

```text
case_study_scores/
```

These files contain ranked candidate drug-target pairs and their predicted
scores. The top-ranked candidates were checked against DrugBank annotations in
the manuscript. The case-study script is:

```text
code/case_study.py
```

## Visualization and Structural Interpretability

UMAP feature visualizations and case-study resources are provided in:

```text
visualization/
```

PDB-related files used for activation-based structural interpretability are
provided in:

```text
PDB_FILES/
```

The structural interpretability analysis compares highlighted ligand atoms with
ligand-protein interaction maps and binding-pocket views from PDB co-crystal
structures.

## Output Files

Training and evaluation scripts write result files to `output/`. Typical result
files include:

```text
result_metrics.pt
config.txt
model_architecture.txt
cross_validation_results_*.csv
```

Large checkpoint files are not required for reproducing the reported metrics and
may be omitted from the review package to reduce storage size.

## Notes for Reproducibility

- The same processed dataset splits should be used when comparing SF-DTI with
  baseline methods.
- The manuscript reports mean and standard deviation across repeated runs or
  cross-validation folds, depending on the evaluation scenario.
- Pretrained encoders are used as frozen feature extractors in the main
  experiments.
- If pre-extracted features are regenerated, minor numerical differences may
  occur because of hardware, library, and precision differences.

## References

1. Chithrananda, S., Grand, G., and Ramsundar, B. ChemBERTa: Large-Scale
   Self-Supervised Pretraining for Molecular Property Prediction. arXiv
   preprint arXiv:2010.09885, 2020.
2. Lin, Z. et al. Evolutionary-scale prediction of atomic-level protein
   structure with a language model. Science, 379, 1123-1130, 2023.
3. Elnaggar, A. et al. ProtTrans: Toward Understanding the Language of Life
   Through Self-Supervised Learning. IEEE Transactions on Pattern Analysis and
   Machine Intelligence, 44, 7112-7127, 2022.
4. Wishart, D. S. et al. DrugBank 5.0: a major update to the DrugBank database
   for 2018. Nucleic Acids Research, 46, D1074-D1082, 2018.
