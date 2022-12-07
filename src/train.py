"""
Training pipeline.
"""
import os
import random
import json

from argparse import ArgumentParser
import torch
import gc
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, \
    EarlyStopping, LearningRateMonitor
from transformers import LayoutLMv2ForTokenClassification

from data.data_module import TicketsDataModule
from mappings import label2id
from utils import get_project_root


def main(args):
    """
    Main method.

    Args:
        args (_type_): _description_
    """
    torch.cuda.empty_cache()
    gc.collect()

    # Choix du nombre de coeur du CPU pour l'entrainement
    gpus = args.gpus
    if gpus == 0:
        cores = os.cpu_count()
        torch.set_num_threads(cores)

    with open(
        os.path.join(get_project_root(), 'data/sample/labeled_sample.json')
    ) as f:
        data = json.load(f)
    random.shuffle(data)

    data_module = TicketsDataModule(
        data=data[:40],
        test_data=data[40:len(data)],
        batch_size=4,
        num_workers=72)  # type: ignore

    model = LayoutLMv2ForTokenClassification.from_pretrained(
        'microsoft/layoutlmv2-base-uncased',
        num_labels=len(label2id)
    )

    EXPERIMENT_NAME = f"{model.__class__.__name__}"
    if args.s3:
        logs_dir = 's3://projet-socratext/logs'
    else:
        logs_dir = 'logs'
    logger = TensorBoardLogger(logs_dir, name=EXPERIMENT_NAME)

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_top_k=1,
        save_last=True,
        mode="min")
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        mode="min",
        patience=40
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
        logger=logger,
        max_epochs=5,
        gpus=gpus,
        num_sanity_val_steps=0
    )
    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--s3', dest='s3', action='store_true')
    parser.add_argument('--lr', default=0.001)
    parser.set_defaults(s3=False)

    args = parser.parse_args()
    main(args)
