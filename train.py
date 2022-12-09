import candle
import os
from train_paccmann2 import main
import json


# This should be set outside as a user environment variable
#os.environ['CANDLE_DATA_DIR'] = '/homes/brettin/Singularity/workspace/data_dir/'
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
        'paccmann_mca_params.txt',
        'pytorch',
        prog='PaccmannMCA_candle',
        desc='Data Preprocessor'
    )
    #Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

def preprocess(params):
    #fname='Data.zip'
    origin=params['data_url']
    # Download and unpack the data in CANDLE_DATA_DIR
    params["train_data"] = candle.get_file(params['train_data'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["val_data"] = candle.get_file(params['val_data'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["gep_filepath"] = candle.get_file(params['gep_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["smi_filepath"] = candle.get_file(params['smi_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["gene_filepath"] = candle.get_file(params['gene_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["smiles_language_filepath"] = candle.get_file(params['smiles_language_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    return params

def run(params):
    params['data_type'] = str(params['data_type'])
    with open ((params['output_dir']+'/params.json'), 'w') as outfile:
        json.dump(params, outfile)
    scores = main(params)
    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print('IMPROVE_RESULT val_loss:\t' + str(scores['val_loss']))

def candle_main():
    params = initialize_parameters()
    params = preprocess(params)
    run(params)

if __name__ == "__main__":
    candle_main()

