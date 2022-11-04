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

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    'train_sensitivity_filepath', type=str,
    help='Path to the drug sensitivity (IC50) data.'
)

parser.add_argument(
    'gep_filepath', type=str,
    help='Path to the gene expression profile data.'
)
parser.add_argument(
    'smi_filepath', type=str,
    help='Path to the SMILES data.'
)
parser.add_argument(
    'gene_filepath', type=str,
    help='Path to a pickle object containing list of genes.'
)
parser.add_argument(
    'smiles_language_filepath', type=str,
    help='Path to a pickle object a SMILES language object.'
)
parser.add_argument(
    'output_dir', type=str,
    help='Directory where the model will be stored.'
)

parser.add_argument(
    'model_name', type=str,
    help='Name for the training.'
)

parser.add_argument(
    'model_params', type=dict,
    help='Dictionary of parameters.'
)
# yapf: enable


def main(
    train_sensitivity_filepath,  gep_filepath,
    smi_filepath, gene_filepath, smiles_language_filepath, output_dir,
    model_name, model_params
):

    logger = logging.getLogger(f'{model_name}')
    # Process parameter file:
    params = model_params

    # Create model directory and dump files
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    #with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
   	 #json.dump(dict(params), fp, indent=4)

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESLanguage.load(smiles_language_filepath)

    # Load the gene list
    with open(gene_filepath, 'rb') as f:
        gene_list = pickle.load(f)

    # Assemble datasets
    train_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=train_sensitivity_filepath,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get('drug_sensitivity_min_max', True),
        drug_sensitivity_processing_parameters=params.get(
            'drug_sensitivity_processing_parameters', {}
        ),
        augment=params.get('augment_smiles', True),
        canonical=params.get('canonical', False),
        kekulize=params.get('kekulize', False),
        all_bonds_explicit=params.get('all_bonds_explicit', False),
        all_hs_explicit=params.get('all_hs_explicit', False),
        randomize=params.get('randomize', False),
        remove_bonddir=params.get('remove_bonddir', False),
        remove_chirality=params.get('remove_chirality', False),
        selfies=params.get('selfies', False),
        add_start_and_stop=params.get('smiles_start_stop_token', True),
        padding_length=params.get('smiles_padding_length', None),
        gene_expression_standardize=params.get(
            'gene_expression_standardize', True
        ),
        gene_expression_min_max=params.get('gene_expression_min_max', False),
        gene_expression_processing_parameters=params.get(
            'gene_expression_processing_parameters', {}
        ),
        device=torch.device(params.get('dataset_device', 'cpu')),
        backend='eager'
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )

    logger.info(
        f'Training dataset has {len(train_dataset)} samples.'
    )

    device = get_device()
    logger.info(
        f'Device for data loader is {train_dataset.device} and for '
        f'model is {device}'
    )
    save_top_model = os.path.join(model_dir, 'weights/{}_{}_{}.pt')
    params.update({  # yapf: disable
        'number_of_genes': len(gene_list),  # yapf: disable
        'smiles_vocabulary_size': smiles_language.number_of_tokens,
        'drug_sensitivity_processing_parameters':
            train_dataset.drug_sensitivity_processing_parameters,
        'gene_expression_processing_parameters':
            train_dataset.gene_expression_dataset.processing
    })

    model = MODEL_FACTORY[params.get('model_fn', 'mca')](params).to(device)
    model._associate_language(smiles_language)

    # Define optimizer
    optimizer = (
        OPTIMIZER_FACTORY[params.get('optimizer', 'Adam')]
        (model.parameters(), lr=params.get('learning_rate', 0.01))
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params.update({'number_of_parameters': num_params})
    logger.info(f'Number of parameters {num_params}')
   
    # Overwrite params.json file with updated parameters.
    #with open(os.path.join(model_dir, 'model_params.json'), 'w') as fp:
        #json.dump(params, fp)

    # Start training
    logger.info('Training about to start...\n')
    t = time()

    model.save(
        save_top_model.format('epoch', '0', params.get('model_fn', 'mca'))
    )
    # Candle checkpointing
    ckpt = candle.CandleCkptPyTorch(params)
    ckpt.set_model({"model": model, "optimizer": optimizer})
    J = ckpt.restart(model)
    if J is not None:
        initial_epoch = J["epoch"]
        print("restarting from ckpt: initial_epoch: %i" % initial_epoch)

    for epoch in range(params['epochs']):

        model.train()
        logger.info(f"== Epoch [{epoch}/{params['epochs']}] ==")
        train_loss = 0

        for ind, (smiles, gep, y) in enumerate(train_loader):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device)
            )
            loss = model.loss(y_hat, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(),1e-6)
            optimizer.step()
            train_loss += loss.item()

        logger.info(
            "\t **** TRAINING ****   "
            f"Epoch [{epoch + 1}/{params['epochs']}], "
            f"loss: {train_loss / len(train_loader):.5f}. "
            f"This took {time() - t:.1f} secs."
        )
        t = time()


        def save(path, metric, typ, val=None):
            model.save(path.format(typ, metric, params.get('model_fn', 'mca')))

    ckpt.ckpt_epoch(epoch, train_loss)
    save(save_top_model, 'training', 'done')
    logger.info('Done with training, models saved, shutting down.')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # run the training
    main(
        args.train_sensitivity_filepath, 
        args.gep_filepath, args.smi_filepath, args.gene_filepath,
        args.smiles_language_filepath, args.output_dir, 
        args.model_name, args.model_params
    )