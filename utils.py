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
from sklearn.model_selection import train_test_split
from data_utils import load_drug_response_data, process_response_data, download_candle_split_data
from data_utils import process_candle_smiles_data, process_candle_gexp_data
# file_path = os.path.dirname(os.path.realpath(__file__))

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def train(params):
    # train_data = params['train_data']
    # val_data = params['val_data']
    gep_filepath = params['gep_filepath']
    smi_filepath = params['smi_filepath']
    gene_filepath = params['gene_filepath']
    smiles_language_filepath = params['smiles_language_filepath']
    output_dir = params['output_dir']
    model_name = params['model_name']
    metric = params['metric']
    data_split_id = params['data_split_id']
   
    
    if params['data_source'] == 'ccle_candle':
        data_type = "CCLE"
    elif params['data_source'] == 'ctrpv2_candle':
        data_type = "CTRPv2"
    elif params['data_source'] == 'gdscv1_candle':
        data_type = "GDSCv1"
    elif params['data_source'] == 'gdscv2_candle':
        data_type = "GDSCv2"
    elif params['data_source'] == 'gsci_candle':
        data_type = "gCSI"
    


    logger = logging.getLogger(f'{model_name}')
    # Create model directory and dump files
    #model_dir = os.path.join(output_dir, model_name)
    model_dir = output_dir
    os.makedirs(os.path.join(model_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)

    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESLanguage.load(smiles_language_filepath)

    # Load the gene list
    with open(gene_filepath, 'rb') as f:
        gene_list = pickle.load(f)

    data_dir = params['CANDLE_DATA_DIR'] + '/'+ params['model_name']+'/Data/'



    # general loading
    train_data = load_drug_response_data(data_path=data_dir, data_type=data_type, split_id=data_split_id,
         split_type='train', response_type=metric, sep="\t", dropna=True)
    val_data = load_drug_response_data(data_path=data_dir, data_type=data_type, split_id=data_split_id,
         split_type='val', response_type=metric, sep="\t", dropna=True)
    test_data = load_drug_response_data(data_path=data_dir, data_type=data_type, split_id=data_split_id,
         split_type='test', response_type=metric, sep="\t", dropna=True)
         
    if params['data_split_seed']>-1:
        all_data = pd.concat([train_data, val_data, test_data], axis=0)
        all_data.reset_index(drop=True, inplace=True)
        # response = pd.read_csv( os.path.join(data_dir, 'response.csv'), index_col=0 )
        train_data, val_data = train_test_split(all_data, test_size=.2, random_state=params['data_split_seed'])
        val_data, test_data = train_test_split(val_data, test_size=.5, random_state=params['data_split_seed'])

  
    # pacmannn specific processing
    train_data = process_response_data(train_data, metric)
    val_data = process_response_data(val_data, metric)
    test_data = process_response_data(test_data, metric)
    process_candle_smiles_data(data_dir)
    process_candle_gexp_data(data_dir)

        
    # Assemble datasets
    train_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=train_data,
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

    val_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=val_data,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get('drug_sensitivity_min_max', True),
        drug_sensitivity_processing_parameters=params.get(
            'drug_sensitivity_processing_parameters', {}
        ),
        augment=params.get('augment_test_smiles', False),
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
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )
    logger.info(
        f'Validation set has '
        f'{len(val_dataset)}.'
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
    
    scores = {}
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

        # Measure validation performance
        model.eval()
        with torch.no_grad():
            val_loss = 0
            predictions = []
            labels = []
            for ind, (smiles, gep, y) in enumerate(val_loader):
                y_hat, pred_dict = model(
                    torch.squeeze(smiles.to(device)), gep.to(device)
                )
                predictions.append(y_hat)
                labels.append(y)
                loss = model.loss(y_hat, y.to(device))
                val_loss += loss.item()

        predictions = np.array(
            [p.cpu() for preds in predictions for p in preds]
        ,dtype = np.float )
        predictions = predictions[0:len(predictions)]
        labels = np.array([l.cpu() for label in labels for l in label],dtype = np.float)
        labels = labels[0:len(labels)]
        val_pearson_a = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
        val_spearman_a = spearmanr(labels, predictions)[0]
        mean_absolute_error = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
        r2 = sklearn.metrics.r2_score(y_true=labels, y_pred=predictions)
        val_rmse_a = np.sqrt(np.mean((predictions - labels)**2))
        val_loss_a = val_loss / len(val_loader)
        if epoch == 0:
            min_val_loss = val_loss_a
            scores['val_loss'] = val_loss_a
            scores['scc'] = val_spearman_a
            scores['pcc'] = val_pearson_a.cpu().detach().numpy().tolist()
            scores['rmse'] = val_rmse_a
        if val_loss_a<min_val_loss:
            min_val_loss = val_loss_a
            #Creating a dictionary with the scores
            #scores['r2'] = r2
            #scores['mean_absolute_error'] = mean_absolute_error
            scores['val_loss'] = val_loss_a
            scores['scc'] = val_spearman_a
            scores['pcc'] = val_pearson_a.cpu().detach().numpy().tolist()
            scores['rmse'] = val_rmse_a
            logger.info(
                f"\t **** VALIDATION  **** "
                f"loss: {val_loss_a:.5f}, "
                f"Pearson: {val_pearson_a:.3f}, "
                f"RMSE: {val_rmse_a:.3f}"
            )
            # Save scores and final preds
            pred = pd.DataFrame({"True": labels, "Pred": predictions}).reset_index()
            te_df1 = val_loader.dataset.drug_sensitivity_df[['drug','cell_line', 'IC50']].reset_index()
            te_df = te_df1.rename(columns={'drug': 'DrugID', 'cell_line': 'CancID', 'IC50':'IC50'})
            pred = pd.concat([te_df, pred], axis=1)
            pred['IC50'] = ((pred['IC50']*1000).apply(np.round))/1000
            pred['True'] = ((pred['True']*1000).apply(np.round))/1000
            pred_fname = str(model_dir+'/results/val_pred.csv')
            pred.to_csv(pred_fname, index=False)

        ckpt.ckpt_epoch(epoch, val_loss_a)

    logger.info('Done with training, models saved, shutting down.')
    return scores, test_data



def predict(
    test_data, gep_filepath,
    smi_filepath, gene_filepath, smiles_language_filepath, output_dir,
    model_name, model_params
):

    logger = logging.getLogger(f'{model_name}')
    # Process parameter file:
    params = model_params

    # Create model directory 
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(os.path.join(model_dir, 'results'), exist_ok=True)


    # Prepare the dataset
    logger.info("Start data preprocessing...")

    # Load SMILES language
    smiles_language = SMILESLanguage.load(smiles_language_filepath)

    # Load the gene list
    with open(gene_filepath, 'rb') as f:
        gene_list = pickle.load(f)


    test_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=test_data,
        smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath,
        smiles_language=smiles_language,
        gene_list=gene_list,
        drug_sensitivity_min_max=params.get('drug_sensitivity_min_max', True),
        drug_sensitivity_processing_parameters=params.get(
            'drug_sensitivity_processing_parameters', {}
        ),
        augment=params.get('augment_test_smiles', False),
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
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=params.get('num_workers', 0)
    )
    logger.info(
        f'test set has '
        f'{len(test_dataset)}.'
    )

    device = get_device()

    model = MODEL_FACTORY[params.get('model_fn', 'mca')](params).to(device)
    model._associate_language(smiles_language)

    # Define optimizer
    optimizer = (
        OPTIMIZER_FACTORY[params.get('optimizer', 'Adam')]
        (model.parameters(), lr=params.get('learning_rate', 0.01))
    )
    # Loading trained model
    checkpoint = torch.load( str(output_dir + '/ckpts/best/model.h5') )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss'] 

    # Measure test performance
    model.eval()
    with torch.no_grad():
        test_loss = 0
        predictions = []
        labels = []
        for ind, (smiles, gep, y) in enumerate(test_loader):
            y_hat, pred_dict = model(
                torch.squeeze(smiles.to(device)), gep.to(device)
            )
            predictions.append(y_hat)
            labels.append(y)
            loss = model.loss(y_hat, y.to(device))
            test_loss += loss.item()

    predictions = np.array(
        [p.cpu() for preds in predictions for p in preds]
    ,dtype = np.float )
    predictions = predictions[0:len(predictions)]
    labels = np.array([l.cpu() for label in labels for l in label],dtype = np.float)
    labels = labels[0:len(labels)]
    test_pearson_a = pearsonr(torch.Tensor(predictions), torch.Tensor(labels))
    test_spearman_a = spearmanr(labels, predictions)[0]
    mean_absolute_error = sklearn.metrics.mean_absolute_error(y_true=labels, y_pred=predictions)
    r2 = sklearn.metrics.r2_score(y_true=labels, y_pred=predictions)
    test_rmse_a = np.sqrt(np.mean((predictions - labels)**2))
    test_loss_a = test_loss / len(test_loader)
    #Creating a dictionary with the scores
    scores = {}
    scores['r2'] = r2
    scores['mean_absolute_error'] = mean_absolute_error
    scores['spearmanr'] = test_spearman_a
    scores['pearsonr'] = test_pearson_a.cpu().detach().numpy().tolist()
    logger.info(
        f"\t **** TESTING  **** "
        f"loss: {test_loss_a:.5f}, "
        f"Pearson: {test_pearson_a:.3f}, "
        f"RMSE: {test_rmse_a:.3f}"
    )
    # Save scores and final preds
    save_dir = str(model_dir+'/results/results.json')
    with open(save_dir, 'w') as fp:
        json.dump(scores, fp)
    pred = pd.DataFrame({"true": labels, "pred": predictions}).reset_index()
    te_df1 = test_loader.dataset.drug_sensitivity_df[['drug','cell_line', 'IC50']].reset_index()
    # te_df = te_df1.rename(columns={'drug': 'DrugID', 'cell_line': 'CancID', 'IC50':'IC50'})
    te_df = te_df1.rename(columns={'drug': 'drug_id', 'cell_line': 'cell_line_id', 'IC50':'labels'})
    pred = pd.concat([te_df, pred], axis=1)
    pred['labels'] = ((pred['labels']*1000).apply(np.round))/1000
    pred['true'] = ((pred['true']*1000).apply(np.round))/1000
    # pred_fname = str(model_dir+'/results/pred.csv')
    pred_fname = os.path.join(output_dir,'test_predictions.csv')
    pred.to_csv(pred_fname, index=False)


