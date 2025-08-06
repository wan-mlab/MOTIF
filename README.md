# temp-TLDA model
## Citation
*“Title of the Paper: A Study using Transfer Learning”*  
_Author1, Author2, et al. (Year)_

## Overview
Racial disparities in breast cancer outcomes remain a critical public health issue, particularly for African American women who experience significantly higher mortality rates compared to European American women. One of the contributing factors is the underrepresentation of racially diverse populations in cancer genomics datasets, which limits the generalizability of machine learning (ML) models. In this study, we propose a weighted multimodal model that integrates transfer learning and data augmentation techniques to address these disparities. Using mRNA, miRNA, and DNA methylation data from The Cancer Genome Atlas (TCGA), we pretrain our model on European American samples and adapt it to African American data through transfer learning. We also apply SMOTE to enhance minority class representation. Additionally, we implement omics-aware weighting to optimize the integration of multi-omics features. Our results demonstrate that the combined application of transfer learning and data augmentation, with appropriate omics weighting, significantly improves predictive performance in the African American cohort. This approach provides a promising strategy to improve model equity and accuracy in cancer prognosis for underrepresented populations.  

<p align="center">
  <img src="FlowChart.png" width="65%"/>
</p>

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

## Requirements
This model requires following python packages. 

| Packages         | Version                       |
|--------------------|----------------------------|
| numpy              | 1.21.6                     |
| pandas             | 1.3.5                      |
| tensorflow         | 2.11.0                     |
| keras              | 2.11.0                     |
| theano             | 1.0.5                      |
| lasagne            | (See installation note below)    |
| scikit-learn       | 1.2.2                      |
| imbalanced-learn   | 0.10.1                     |
| scipy              | 1.10.1                     |

🔔 **Note**: `lasagne` is not available on PyPI. Install via GitHub:  
> `pip install https://github.com/Lasagne/Lasagne/archive/master.zip`

## Installation
1. Clone the temp-TLDA_Model git repository
```bash
git clone https://github.com/wan-mlab/temp-TLDA_Model.git
```
2. Navigate to the directory of temp-TLDA_Model package
```bash
cd /your path/temp-TLDA_Model
```

## Instructions

```
usage: BRCA_weightedTLDA.py [-h] --in_path IN_PATH --out_path OUT_PATH --cpoint CPOINT --year YEAR

This script implements a fairness-aware deep learning model designed to mitigate
racial disparities in breast cancer prognosis. It integrates transfer learning and
synthetic data augmentation (SMOTE) with weighted multi-omics representation to
improve predictive performance in underrepresented populations. The model is
pretrained on European American data and fine-tuned on African American data using
contrastive domain adaptation.

optional arguments:
  -h, --help            show this help message and exit
  --in_path IN_PATH     Path to the input multi-omics data and other materials
                        for model training.
  --out_path OUT_PATH   Path for all output files after model training.
  --cpoint CPOINT       Clinical endpoint of interest for prognosis prediction
                        (e.g., PFI, DSS).
  --year YEAR           Time threshold used to define positive prognosis events.

Examples:
  python BRCA_weightedTLDA.py \
      --in_path /home/user/project/input \
      --out_path /home/user/project/output \
      --cpoint PFI \
      --year 3
```
  
## Troubleshooting 

We kindly ask users and contributors to utilize the Issues section of this repository for reporting bugs, requesting new features, or documenting any technical difficulties encountered during usage. We monitor submitted issues regularly and will do our best to respond in a timely and constructive manner. 

