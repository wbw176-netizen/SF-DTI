# 📊 Datasets Distribution Analysis

> 🔍 Exploratory analysis for understanding dataset heterogeneity across multiple DTI benchmarks.

---

## 🌟 Overview

Before model training and evaluation, it is useful to understand **how different the datasets actually are**.  
This module performs a **distribution analysis** over six benchmark DTI datasets:

- 🧪 **BindingDB**
- 🧬 **Biosnap**
- 🐛 **Celegans**
- 💊 **Drugbank**
- 👤 **Human**
- 📈 **KIBA**

This analysis provides an intuitive view of:

- 📦 dataset scale
- 🧩 entity coverage
- 🧪 drug chemical-space distribution
- 🧬 protein sequence-space distribution
- 📏 SMILES/protein length differences

These observations are helpful for interpreting model behavior under **random split**, **cold-start**, and **cross-domain** settings.

---

## 🧭 What This Experiment Analyzes

### 1️⃣ Basic dataset statistics

For each dataset, we summarize:

- number of drug–target pairs
- number of positive / negative samples
- number of unique drugs
- number of unique targets
- average and median SMILES length
- average and median protein length

### 2️⃣ Drug chemical space

For each unique drug:

- convert `SMILES` into **Morgan fingerprints** using RDKit
- reduce the high-dimensional fingerprints into 2D using **UMAP**
- color points by dataset

This helps visualize whether different datasets cover similar or distinct regions in chemical space.

### 3️⃣ Protein sequence space

For each unique target protein:

- represent the sequence using **3-mer TF-IDF**
- compress features with **Truncated SVD**
- project them to 2D using **UMAP**

This helps reveal potential target-space shifts across datasets.

### 4️⃣ Length distributions

We also plot:

- 📏 **SMILES length distribution**
- 📏 **Protein length distribution**

These plots help identify differences in sequence/structure complexity across datasets.
