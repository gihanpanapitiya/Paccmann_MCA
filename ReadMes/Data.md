Cancer drug response prediction models typically use a variety of input data including drug features, cell-line features as well as the response data. 

As part of model curation, the original data that is provided by the authors is copied to an FTP site. The data used for this model curation can be downloaded from [here](https://ftp.mcs.anl.gov/pub/candle/public/improve/Paccmann_MCA).

## Data
1) 2128_genes.pkl - Gene list considered for the experiments 
2) gdsc_cell_line_ic50_test_fraction_0.1_id_997_seed_42.csv - Test response data
3) gdsc_cell_line_ic50_train_fraction_0.9_id_997_seed_42.csv - Train response data
5) gdsc-rnaseq_gene-expression.csv - Gene expression data
6) gdsc.smi - SMILES data in .smi format
7) smiles_language_chembl_gdsc_ccle.pkl - SMILESLanguage generated from the smi file 

All this data was provided by the author and downloaded from [here](https://ibm.ent.box.com/v/paccmann-pytoda-data). 

All data preprocessing was done within the main script using the [Pytoda](https://paccmann.github.io/paccmann_datasets/api/pytoda.html) package for handling both SMILES and gene expression data. There is no separate preprocessing script to handle the author provided data.

## Using your own data
Ultimately, we want to be able to train models with other datasets (not only the ones provided with the model repo). 
This requires the preprocessing scripts to be available and reproducible.


