import os
import pandas as pd

def process_response_data(df_resp, response_type='ic50'):    
    # df = pd.read_csv('response.tsv', sep='\t')
    drd = df_resp[['improve_chem_id', 'improve_sample_id', response_type]]
    drd.columns =['drug','cell_line','IC50']
    # drd = drd.dropna(axis=0)
    drd.reset_index(drop=True, inplace=True)
    # drd.to_csv('tmp/Paccmann_MCA/Data/response.csv')
    return drd

def process_candle_smiles_data(smiles_df, data_dir):
    # smiles_name = 'drug_SMILES.tsv'
    # if self.version=='benchmark-data-imp-2023':
    #     smiles_name = 'drug_SMILES.txt'

    # smiles = pd.read_csv( os.path.join(data_dir, smiles_name), sep='\t')
    smiles = smiles_df[['smiles', 'improve_chem_id']]
    smiles.to_csv(data_dir+'/candle_smiles.smi', header=None, index=False, sep='\t')


def process_candle_gexp_data(gexp, data_dir):

    gexp.index.name=None
    gexp.to_csv(data_dir+'/candle_gexp.csv')
