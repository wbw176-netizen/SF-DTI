# SF-DTI


## 🧬 Feature Extraction with Pre-trained Language Models (PLMs)

We leverage three state-of-the-art Pre-trained Language Models (PLMs) to extract comprehensive semantic features for drugs and targets. The feature extraction pipeline supports variable-length storage with `float16` precision to optimize computational efficiency.

### 1. Supported Models & Download Links
Please download the model weights from Hugging Face and place them in your local directory (e.g., `./plm_models/`).

| Modality | Model Name | Description | Hugging Face Link |
| :--- | :--- | :--- | :--- |
| **Drug** | **ChemBERTa** | Pre-trained on 77M SMILES strings (DeepChem). | [DeepChem/ChemBERTa-77M-MLM](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM) |
| **Target** | **ESM-2** | Evolutionary Scale Modeling (3B parameters). | [facebook/esm2_t36_3B_UR50D](https://huggingface.co/facebook/esm2_t36_3B_UR50D) |
| **Target** | **ProtT5** | T5-based model trained on UniRef50 (XL). | [Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) |

### 2. Usage

We provide the `start_extracting.py` script to handle feature extraction. You can run it in **single mode** (for one dataset) or **batch mode** (for multiple random splits).

#### 🔧 Preparation
Ensure your model directories are organized. For example:
```bash
./plm_models/
├── chemberta_model/  # Model files from DeepChem
├── esm2_3B_model/    # Model files from Facebook
└── protT5_model/     # Model files from Rostlab
▶️ Option A: Extract for a Single Dataset
To extract features for a specific dataset (must contain train.csv, val.csv, test.csv):
python start_extracting.py single \
  --data_dir ./datasets/Drugbank/random2 \
  --output_dir ./code/features/varlen \
  --esm2_dir ./plm_models/esm2_3B_model \
  --chemberta_dir ./plm_models/chemberta_model \
  --prott5_dir ./plm_models/protT5_model \
  --batch_size 4
🔁 Option B: Batch Extraction
To batch process multiple datasets (iterating through random2 subfolders):
python start_extracting.py batch \
  --data_dir ./datasets \
  --output_dir ./code/features/varlen \
  --esm2_dir ./plm_models/esm2_3B_model \
  --chemberta_dir ./plm_models/chemberta_model \
  --prott5_dir ./plm_models/protT5_model
```
### 3. References

If you use these pre-trained models or our feature extraction pipeline, please cite the original papers:

1. **ChemBERTa**: Chithrananda, S., Grand, G., & Ramsundar, B. (2020). *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction*. arXiv preprint arXiv:2010.09885.
2. **ESM-2**: Lin, Z., Akin, H., Rao, R., et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model*. Science, 379(6637), 1123-1130.
3. **ProtT5**: Elnaggar, A., Heinzinger, M., Dallago, C., et al. (2021). *ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 7112-7127.
