import os

import comet_ml
from pytorch_lightning.loggers import CometLogger
import torch
import lightning as L
import copy
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from .distiler import Distiler
from .prepare_data_loader import prepare_data_loader
from .prepare_unsupervised_data_loader import prepare_unsupervised_data_loader

def distil(
    model,
    compiled_model,
    tokenizer,
    translator,
    data,
    unsupervised_data,
    natural_data,
    params=None,
):
    model_copy = copy.deepcopy(model)
    logger = CometLogger(
        api_key= os.getenv('COMET_API_KEY'),
        project_name='distil',
        workspace=os.getenv('COMET_WORKSPACE'),
    )

    first_gpu = 'cuda:0'
    model.to(first_gpu)
    compiled_model.to(first_gpu)
    model_copy.to(first_gpu)

    lightning_model = Distiler(
        model,
        compiled_model,
        model_copy,
        translator,
        params.natural_data_loss,
        detach_and_roll=params.detach_and_roll,
        algorithm_loss=params.algorithm_loss,
        unsupervised_loss=params.unsupervised_loss,
        pad_communication=params.pad_communication,
        gate_dims_to_final_result=params.only_result_subspace,
        algorithm_loss_multiplier=params.algorithm_loss_multiplier,
    )
    data_loader, val_dataloader = prepare_data_loader(
        data,
        tokenizer,
        max_length=params.max_sequence_length_supervised,
        batch_size=params.batch_size
    )
    unsupervised_data_loader = prepare_unsupervised_data_loader(
        unsupervised_data,
        tokenizer,
        max_length=params.max_sequence_length,
        batch_size=params.batch_size
    )

    combined_dataloaders = (data_loader, unsupervised_data_loader)
    combined_eval_dataloaders = (val_dataloader,)

    trainer = L.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        max_epochs=1,
        limit_train_batches=params.train_batches,
        val_check_interval=100,
        precision=32,
    )

    trainer.fit(
        lightning_model,
        train_dataloaders=combined_dataloaders,
        val_dataloaders=combined_eval_dataloaders
    )

    return lightning_model.adapter
