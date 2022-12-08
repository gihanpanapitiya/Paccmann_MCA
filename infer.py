import candle
import os
from test_paccmann import main


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
    fname='Data.zip'
    origin=params['data_url']
    # Download and unpack the data in CANDLE_DATA_DIR
    candle.file_utils.get_file(fname, origin)
    # Model-specific changes
    params['test_data'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['test_data']
    params['gep_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['gep_filepath']
    params['smi_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['smi_filepath']
    params['gene_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['gene_filepath']
    params['smiles_language_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/common/Data/'+params['smiles_language_filepath']
    return params


def run(params):
    keys_parsing = ["train_data", "test_data", "val_data",
        "gep_filepath", "smi_filepath", "gene_filepath",
        "smiles_language_filepath", "output_dir",
        "model_name"]
    model_param_key = []
    for key in params.keys():
    	if key not in keys_parsing:
        	model_param_key.append(key)
    model_params = {key: params[key] for key in model_param_key}
    params['model_params'] = model_params
    args = candle.ArgumentStruct(**params)
    main(args.test_data,
        args.gep_filepath, args.smi_filepath, args.gene_filepath,
        args.smiles_language_filepath, args.output_dir,
        args.model_name, args.model_params)

def candle_main():
    params = initialize_parameters()
    params = preprocess(params)
    run(params)
    
if __name__ == "__main__":
    candle_main()

