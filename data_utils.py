import os
from typing import Union, List
import urllib
import pandas as pd
#------------------
# 1. download data
#------------------
version = 'benchmark-data-pilot1'
ftp_dir = 'https://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/raw_data/'
candle_data_dict = {
    'ccle_candle': "CCLE",
    'ctrpv2_candle':"CTRPv2",
    'gdscv1_candle':"GDSCv1",
    'gdscv2_candle':"GDSCv2",
    'gsci_candle': "gCSI"}

def download_candle_split_data(data_type="CCLE", split_id=0, data_dest='Data/'):
    print(f'downloading {data_type} split {split_id} data')

    
    split_src = os.path.join(ftp_dir, 'splits')
    train_split_name = f'{data_type}_split_{split_id}_train.txt'
    val_split_name = f'{data_type}_split_{split_id}_val.txt'
    test_split_name = f'{data_type}_split_{split_id}_test.txt'

    
    # download split data
    for file in [train_split_name, val_split_name, test_split_name]:
        src = os.path.join(split_src, file)
        dest = os.path.join(data_dest, file)
        urllib.request.urlretrieve(src, dest)

def download_candle_resp_data(data_dest='Data/'):


    print('downloading response data')
    # download response data
    resp_name = 'response.tsv'
    src = os.path.join(ftp_dir, 'y_data', resp_name)
    dest = os.path.join(data_dest, resp_name)
    urllib.request.urlretrieve(src, dest)

def download_candle_gexp_data(data_dest='Data/'):

    # gene expression data
    print('downloading expression data')
    gexp_name = 'cancer_gene_expression.tsv'
    src = os.path.join(ftp_dir, 'x_data', gexp_name)
    dest = os.path.join(data_dest, gexp_name)
    urllib.request.urlretrieve(src, dest)

def download_candle_mut_data(data_dest='Data/'):
    # gene mutation data
    print('downloading mutation data')
    gmut_name = 'cancer_mutation_count.tsv'
    src = os.path.join(ftp_dir, 'x_data', gmut_name)
    dest = os.path.join(data_dest, gmut_name)
    urllib.request.urlretrieve(src, dest)

def download_candle_smiles_data(data_dest='Data/'):
    # gene mutation data
    print('downloading smiles data')
    if version == 'benchmark-data-pilot1':
        smiles_name = 'drug_SMILES.tsv'
    src = os.path.join(ftp_dir, 'x_data', smiles_name)
    dest = os.path.join(data_dest, smiles_name)
    urllib.request.urlretrieve(src, dest)

def download_candle_data(data_type="CCLE", split_id=0, data_dest='Data/'):
    download_candle_split_data(data_type=data_type, split_id=split_id, data_dest=data_dest)
    download_candle_resp_data(data_dest=data_dest)
    download_candle_gexp_data(data_dest=data_dest)
    download_candle_mut_data(data_dest=data_dest)
    download_candle_smiles_data(data_dest=data_dest)


def load_drug_response_data(data_path, data_type="CCLE",
 split_id=0, split_type='train', response_type='ic50', sep="\t",
 dropna=True):
    """
    Returns datarame with cancer ids, drug ids, and drug response values. Samples
    from the original drug response file are filtered based on the specified
    sources.

    Args:
        source (str or list of str): DRP source name (str) or multiple sources (list of strings)
        split(int or None): split id (int), None (load all samples)
        split_type (str or None): one of the following: 'train', 'val', 'test'
        y_col_name (str): name of drug response measure/score (e.g., AUC, IC50)

    Returns:
        pd.Dataframe: dataframe that contains drug response values
    """
    # TODO: at this point, this func implements the loading a single source
    y_file_path = os.path.join(data_path, 'response.tsv')
    df = pd.read_csv(y_file_path, sep=sep)

    # import pdb; pdb.set_trace()
    if isinstance(split_id, int):
        # Get a subset of samples
        ids = load_split_file(data_path, data_type, split_id, split_type)
        df = df.loc[ids]
    else:
        # Get the full dataset for a given source
        df = df[df["source"].isin([data_type])]

    cols = ["source",
            "improve_chem_id",
            "improve_sample_id",
            response_type]
    df = df[cols]  # [source, drug id, cancer id, response]
    if dropna:
        df.dropna(axis=0, inplace=True)
    df = df.reset_index(drop=True)
    return df


def load_split_file(
    data_path: str,
    data_type: str,
    split_id: Union[int, None]=None,
    split_type: Union[str, List[str], None]=None) -> list:
    """
    Args:
        source (str): DRP source name (str)

    Returns:
        ids (list): list of id integers
    """
    if isinstance(split_type, str):
        split_type = [split_type]

    # Check if the split file exists and load
    ids = []
    for st in split_type:
        fpath = os.path.join(data_path, f"{data_type}_split_{split_id}_{st}.txt")
        # assert fpath.exists(), f"Splits file not found: {fpath}"
        ids_ = pd.read_csv(fpath, header=None)[0].tolist()
        ids.extend(ids_)
    return ids

#-----------------------------------
# 2. preprocess data to swnet format
#-----------------------------------
def process_response_data(df_resp, response_type='ic50'):    
    # df = pd.read_csv('response.tsv', sep='\t')
    drd = df_resp[['improve_chem_id', 'improve_sample_id', response_type]]
    drd.columns =['drug','cell_line','IC50']
    # drd = drd.dropna(axis=0)
    drd.reset_index(drop=True, inplace=True)
    # drd.to_csv('tmp/Paccmann_MCA/Data/response.csv')
    return drd

def process_candle_smiles_data(data_dir):
    smiles = pd.read_csv(data_dir+'/drug_SMILES.tsv', sep='\t')
    smiles = smiles[['canSMILES', 'improve_chem_id']]
    smiles.to_csv(data_dir+'/candle_smiles.smi', header=None, index=False, sep='\t')
# process_candle_smiles_data()


def process_candle_gexp_data(data_dir):

    from typing import List, Union
    def set_col_names_in_multilevel_dataframe(
        df: pd.DataFrame,
        level_map: dict,
        gene_system_identifier: Union[str, List[str]]="Gene_Symbol") -> pd.DataFrame:
        """ Util function that supports loading of the omic data files.
        Returns the input dataframe with the multi-level column names renamed as
        specified by the gene_system_identifier arg.

        Args:
            df (pd.DataFrame): omics dataframe
            level_map (dict): encodes the column level and the corresponding identifier systems
            gene_system_identifier (str or list of str): gene identifier system to use
                options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                        combination of ["Entrez", "Gene_Symbol", "Ensembl"]

        Returns:
            pd.DataFrame: the input dataframe with the specified multi-level column names
        """
        df = df.copy()

        level_names = list(level_map.keys())
        level_values = list(level_map.values())
        n_levels = len(level_names)
        
        if isinstance(gene_system_identifier, list) and len(gene_system_identifier) == 1:
            gene_system_identifier = gene_system_identifier[0]

        # print(gene_system_identifier)
        # import pdb; pdb.set_trace()
        if isinstance(gene_system_identifier, str):
            if gene_system_identifier == "all":
                df.columns = df.columns.rename(level_names, level=level_values)  # assign multi-level col names
            else:
                df.columns = df.columns.get_level_values(level_map[gene_system_identifier])  # retian specific column level
        else:
            assert len(gene_system_identifier) <= n_levels, f"'gene_system_identifier' can't contain more than {n_levels} items."
            set_diff = list(set(gene_system_identifier).difference(set(level_names)))
            assert len(set_diff) == 0, f"Passed unknown gene identifiers: {set_diff}"
            kk = {i: level_map[i] for i in level_map if i in gene_system_identifier}
            # print(list(kk.keys()))
            # print(list(kk.values()))
            df.columns = df.columns.rename(list(kk.keys()), level=kk.values())  # assign multi-level col names
            drop_levels = list(set(level_map.values()).difference(set(kk.values())))
            df = df.droplevel(level=drop_levels, axis=1)
        return df
    def load_gene_expression_data(gene_expression_file_path,
        gene_system_identifier: Union[str, List[str]]="Gene_Symbol",
        sep: str="\t",
        verbose: bool=True) -> pd.DataFrame:
        """
        Returns gene expression data.

        Args:
            gene_system_identifier (str or list of str): gene identifier system to use
                options: "Entrez", "Gene_Symbol", "Ensembl", "all", or any list
                        combination of ["Entrez", "Gene_Symbol", "Ensembl"]

        Returns:
            pd.DataFrame: dataframe with the omic data
        """

        improve_globals_canc_col_name = "improve_sample_id"
        # level_map encodes the relationship btw the column and gene identifier system
        level_map = {"Ensembl": 0, "Entrez": 1, "Gene_Symbol": 2}
        header = [i for i in range(len(level_map))]

        df = pd.read_csv(gene_expression_file_path, sep=sep, index_col=0, header=header)

        df.index.name = improve_globals_canc_col_name  # assign index name
        df = set_col_names_in_multilevel_dataframe(df, level_map, gene_system_identifier)
        if verbose:
            print(f"Gene expression data: {df.shape}")
            # print(df.dtypes)
            # print(df.dtypes.value_counts())
        return df
        
    gexp = load_gene_expression_data('cancer_gene_expression.tsv', gene_system_identifier='Gene_Symbol')
    gexp.index.name=None
    gexp.to_csv(data_dir+'/candle_gexp.csv')

# process_candle_gexp_data()

