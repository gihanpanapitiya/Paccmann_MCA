# Paccmann-MCA
Paccmann-MCA is a multimodal attention-based convolutional encoder based model for cancer drug response prediction

## Structure
This model uses SMILES strings as drug features and gene expression as cell-line features. The drug sensitivity data is obtained from the publicly available Genomics of Drug Sensitivity in Cancer (GDSC) database. A subset of genes are selected using the STRING protein-protein interaction (PPI) network. The gene expression data with the selected genes are then passed to a gene expression encoder to a fully connected dense layer. The SMILES strings are passed through a SMILES encoder, which is a multiscale convolutional attention module that takes into account the SMILES embeddings along with their postional information. The final dense layer combines the outputs from the SMILES encoder and the gene expression encoder to predict the IC50 values.

## Data sources
The primary data sources for this model are:
1) Raw cell line data from GDSC
2) The compound structural information from PubChem and the LINCS database

## Preprocessing
The authors utilize the [Pytoda](https://paccmann.github.io/paccmann_datasets/api/pytoda.html) package developed by the group to handle the preprocessing of the SMILES data.

## Evaluation
Two evaluation schemes were used for the analysis of prediction performance.
1) Strict split - Neither the cell lines nor the compound structures within the test dataset have been seen by the model prior to testing. 
2) Linient split - Instead of fully depriving the model from both the cells and drugs in the test set, this strategy ensured that no cell−drug pair in the test set has been seen before by the model.

## URLs
- Original GitHub: https://github.com/PaccMann/paccmann_predictor
- IMPROVE GitHub: https://github.com/JDACS4C-IMPROVE/Paccmann_MCA/tree/develop
- Data: https://ftp.mcs.anl.gov/pub/candle/public/improve/Paccmann_MCA/Data.zip

## Reference
Manica, Matteo, et al. "Toward explainable anticancer compound sensitivity prediction via multimodal attention-based convolutional encoders." Molecular Pharmaceutics 16.12 (2019): 4797-4806.
