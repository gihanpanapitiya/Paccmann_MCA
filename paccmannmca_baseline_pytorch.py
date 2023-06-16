#!/usr/bin/env python3
"""Train PaccMann predictor."""
import argparse
import json
import logging
import os
import pickle
import sys
from time import time
import numpy as np
import torch
from pytoda.datasets import DrugSensitivityDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from paccmann_predictor.models import MODEL_FACTORY
from paccmann_predictor.utils.hyperparams import OPTIMIZER_FACTORY
from paccmann_predictor.utils.loss_functions import pearsonr
from paccmann_predictor.utils.utils import get_device
import candle
import pandas as pd
from scipy.stats import spearmanr
import sklearn

from utils import train, predict

# setup logging

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
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(preprocessor_bmk)
    return gParameters

def preprocess(params):
    params['train_data'] = os.environ['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'+params['train_data']
    params['val_data'] = os.environ['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'+params['val_data']
    params['gep_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'+params['gep_filepath']
    params['smi_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'+params['smi_filepath']
    params['gene_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'+params['gene_filepath']
    params['smiles_language_filepath'] = os.environ['CANDLE_DATA_DIR'] + '/' +  params['model_name']+'/Data/'+params['smiles_language_filepath']

    params['test_data'] = os.environ['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'+params['test_data']


    """ 
    params["train_data"] = candle.get_file(params['train_data'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["val_data"] = candle.get_file(params['val_data'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["gep_filepath"] = candle.get_file(params['gep_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["smi_filepath"] = candle.get_file(params['smi_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["gene_filepath"] = candle.get_file(params['gene_filepath'], origin, datadir=params['data_dir'], cache_subdir=None)
    params["smiles_language_filepath"] = candle.get_file(params['smiles_language_filepath'], origin, datadir=params['data_dir'], cache_subdir=None) """
    return params



if __name__ == '__main__':
    params = initialize_parameters()
    params = preprocess(params)

    params['data_type'] = str(params['data_type'])
    with open ((params['output_dir']+'/params.json'), 'w') as outfile:
        json.dump(params, outfile)

    # train model
    scores = train(params)


    with open(params['output_dir'] + "/scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    #print('IMPROVE_RESULT RMSE:\t' + str(scores['rmse']))
    print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["val_loss"]))

    # predict
    # args = candle.ArgumentStruct(**params)
    print("output: ", params['output_dir'])
    
    predict(params['test_data'],
    params['gep_filepath'], params['smi_filepath'], params['gene_filepath'],
    params['smiles_language_filepath'], params['output_dir'],
    params['model_name'], params)




    # run(params)

    # main(params)
