import candle
import os
from paccmannmca_baseline_pytorch import main
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from scipy import stats
from typing import List, Union, Optional
import shutil
import improve_utils
from improve_utils import improve_globals as ig


# Currently this code requires CSG data to present in this directory
# In addition to gene expression and smiles, MCA need two other .pkl files for execution
# preprocess.sh need to be modified (change name to preprocess_CSA instead of preprocess) to run this script
#TODO: implement CANDLE get file function to download data from FTP


file_path = os.path.dirname(os.path.realpath(__file__))

additional_definitions = [
    {'name': 'gep_filepath',
     'type': str,
     'help': 'Path to the gene expression profile data.'
     },
    {'name': 'smi_filepath',
     'type': str,
     'help': 'Path to the SMILES data.'
     },
    {'name': 'gene_filepath',
     'type': str,
     'help': 'Path to a pickle object containing list of genes.'
     },
    {'name': 'smiles_language_filepath',
     'type': str,
     'help': 'Path to a pickle object a SMILES language object.'
     },

    {'name': 'drug_sensitivity_min_max',
     'type': bool,
     'help': '.....'
     },
    {'name': 'gene_expression_standardize',
     'type': bool,
     'help': 'Do you want to standardize gene expression data?'
     },
    {'name': 'augment_smiles',
     'type': bool,
     'help': 'Do you want to augment smiles data?'
     },
    {'name': 'smiles_start_stop_token',
     'type': bool,
     'help': '.....'
     },
    {'name': 'number_of_genes',
     'type': int,
     'help': 'Number of selected genes'
     },
    {'name': 'smiles_padding_length',
     'type': int,
     'help': 'Padding length for smiles strings'
     },
    {'name': 'filters',
     'type': list,
     'help': 'Size of filters'
     },
    {'name': 'multiheads',
     'type': list,
     'help': 'Size of multiheads for attention layer'
     },
    {'name': 'smiles_embedding_size',
     'type': int,
     'help': 'Size of smiles embedding'
     },
    {'name': 'kernel_sizes',
     'type': list,
     'help': 'Size of the kernels'
     },
    {'name': 'smiles_attention_size',
     'type': int,
     'help': 'Size of smiles attention'
     },
    {'name': 'embed_scale_grad',
     'type': bool,
     'help': '.....'
     },
    {'name': 'final_activation',
     'type': bool,
     'help': 'Is there a final activation?'
     },
    {'name': 'gene_to_dense',
     'type': bool,
     'help': '.....'
     },
    {'name': 'smiles_vocabulary_size',
     'type': int,
     'help': 'Size of smiles vocabulary'
     },
    {'name': 'number_of_parameters',
     'type': int,
     'help': 'Number of parameters'
     },
    {'name': 'drug_sensitivity_processing_parameters',
     'type': dict,
     'help': 'Parameters for drug sensitivity processing'
     },
    {'name': 'gene_expression_processing_parameters',
     'type': dict,
     'help': 'Parameters for processing gene expression data'
     }
]

required = None


# experimental
supported_definitions = ['data_url','train_data','val_data','shuffle','feature_subsample']


class PaccmannMCA_candle(candle.Benchmark):

    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    preprocessor_bmk = PaccmannMCA_candle(file_path,
        'paccmannmca_default_model.txt',
        'pytorch',
        prog='PaccmannMCA_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

def preprocess(params):
    # Settings:
    y_col_name = "auc"
    import pdb; pdb.set_trace()
    split = 0
    source_data_name = "CCLE"

    # Load train
    rs_tr = improve_utils.load_single_drug_response_data_v2(
        source=source_data_name,
        split_file_name=f"{source_data_name}_split_{split}_train.txt",
        y_col_name=y_col_name)

    # Load val
    rs_vl = improve_utils.load_single_drug_response_data_v2(
        source=source_data_name,
        split_file_name=f"{source_data_name}_split_{split}_val.txt",
        y_col_name=y_col_name)

    # Load test
    rs_te = improve_utils.load_single_drug_response_data_v2(
        source=source_data_name,
        split_file_name=f"{source_data_name}_split_{split}_test.txt",
        y_col_name=y_col_name)

    print("\nResponse train data", rs_tr.shape)
    print("Response val data", rs_vl.shape)
    print("Response test data", rs_te.shape)
    # Load omic feature data
    ge = improve_utils.load_gene_expression_data(gene_system_identifier="Gene_Symbol")
    # Load drug feature data
    sm = improve_utils.load_smiles_data()
    sm = sm.set_index(['improve_chem_id'])
    print(f"Total unique cells: {rs_tr[ig.canc_col_name].nunique()}")
    print(f"Total unique drugs: {rs_tr[ig.drug_col_name].nunique()}")
    assert len(set(rs_tr[ig.canc_col_name]).intersection(set(ge.index))) == rs_tr[ig.canc_col_name].nunique(), "Something is missing..."
    assert len(set(rs_tr[ig.drug_col_name]).intersection(set(sm.index))) == rs_tr[ig.drug_col_name].nunique(), "Something is missing..."
    
    # Modify files to be compatible with Paccmann_MCA
    #smiles
    if not os.path.isfile(file_path+'/candle_data_dir/Data/smiles.smi'):
        sm_new = pd.DataFrame(columns = ['SMILES', 'DrugID'])
        sm_new['SMILES'] = sm['smiles'].values
        sm_new['DrugID'] = sm.index.values
        #sm_new.to_csv(str(os.environ['CANDLE_DATA_DIR']+'/smiles.smi'), index=False)
        sm_new.to_csv(str(file_path+'/candle_data_dir/Data/smiles.csv'), index=False)

        # save smiles as .smi format as required by the code
        newfile = str(file_path+'/candle_data_dir/Data/smiles.smi')
        file = str(file_path+'/candle_data_dir/Data/smiles.csv')
        with open(file,'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  ## skip one line (the first one)
            for line in csv_reader:
                with open(newfile, 'a') as new_txt:    #new file has .txt extn
                    txt_writer = csv.writer(new_txt, delimiter = '\t') #writefile
                    txt_writer.writerow(line)   #write the lines to file`

    #response data
    rs_tr = rs_tr.drop(columns = ['source'])
    rs_tr = rs_tr.rename(columns = {'improve_chem_id':'drug', 'improve_sample_id':'cell_line', 'auc':'IC50'})
    #rs_tr.to_csv(str(os.environ['CANDLE_DATA_DIR']+'/'+params['train_data']))
    rs_tr.to_csv(str(file_path+'/candle_data_dir/Data/train.csv'))

    rs_vl = rs_vl.drop(columns = ['source'])
    rs_vl = rs_vl.rename(columns = {'improve_chem_id':'drug', 'improve_sample_id':'cell_line', 'auc':'IC50'})
    #rs_vl.to_csv(str(os.environ['CANDLE_DATA_DIR']+'/'+params['val_data']))
    rs_vl.to_csv(str(file_path+'/candle_data_dir/Data/val.csv'))

    rs_te = rs_te.drop(columns = ['source'])
    rs_te = rs_te.rename(columns = {'improve_chem_id':'drug', 'improve_sample_id':'cell_line', 'auc':'IC50'})
    #rs_te.to_csv(str(os.environ['CANDLE_DATA_DIR']+'/'+params['test_data']))
    rs_te.to_csv(str(file_path+'/candle_data_dir/Data/test.csv'))
    
    if not os.path.isfile(file_path+'/candle_data_dir/Data/gene_expression.csv'):
        #gene expression
        ge.index.name = 'CancID'
        #ge.to_csv(str(os.environ['CANDLE_DATA_DIR']+'/gene_expression.csv'))
        ge.to_csv(str(file_path+'/candle_data_dir/Data/gene_expression.csv'))

    #Other files needed for Paccmann_MCA
    shutil.copy(os.path.join(file_path,'csa_data','raw_data','x_data','2128_genes.pkl'),os.path.join(file_path,'candle_data_dir','Data','2128_genes.pkl') )
    shutil.copy(os.path.join(file_path,'csa_data','raw_data','x_data','smiles_language_chembl_gdsc_ccle.pkl'),os.path.join(file_path,'candle_data_dir','Data','smiles_language_chembl_gdsc_ccle.pkl') )
    #shutil.copy(os.path.join(file_path,'csa_data','raw_data','x_data','2128_genes.pkl'),os.path.join(os.environ['CANDLE_DATA_DIR'],params['gene_filepath']) )
    #shutil.copy(os.path.join(file_path,'csa_data','raw_data','x_data','smiles_language_chembl_gdsc_ccle.pkl'),os.path.join(os.environ['CANDLE_DATA_DIR'],params['smiles_language_filepath']) )

def candle_main():
    params = initialize_parameters()
    preprocess(params)

if __name__ == "__main__":
    candle_main()

