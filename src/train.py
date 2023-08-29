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
import mlflow
from pytorch_lightning.callbacks import ModelCheckpoint, \
    EarlyStopping, LearningRateMonitor
from transformers import LayoutLMv2Processor, LayoutXLMTokenizerFast, LayoutXLMProcessor, LayoutLMv2FeatureExtractor

from data.data_module import TicketsDataModule
from models.layout_xlm import LayoutXLMModule
from utils import get_project_root


def main(args):
    """
    Main method.

    Args:
        args: Main method arguments.
    """
    torch.cuda.empty_cache()
    gc.collect()

    # Training specs
    gpus = args.gpus
    cores = os.cpu_count()
    if gpus == 0:
        torch.set_num_threads(cores)
        print(f"Using {cores} cpus.")

    # Loading a data sample
    with open(
        os.path.join(get_project_root(), 'data/sample/labeled_sample.json')
    ) as f:
        data = json.load(f)
    # Splitting the data into train/test
    n_samples = len(data)
    random.shuffle(data)
    train_slice = slice(0, int(n_samples * 0.8))
    test_slide = slice(int(n_samples * 0.8), n_samples)

    # Define DataModule
    # processor = LayoutLMv2Processor.from_pretrained(
    #     "microsoft/layoutlmv2-base-uncased",
    #     revision="no_ocr"
    # )
    tokenizer = LayoutXLMTokenizerFast.from_pretrained(
        "microsoft/layoutxlm-base"
        )
    processor = LayoutXLMProcessor(
        LayoutLMv2FeatureExtractor(apply_ocr=False),
        tokenizer
    )

    data_module = TicketsDataModule(
        data=data[train_slice],
        test_data=data[test_slide],
        processor=processor,
        batch_size=int(args.batch_size),
        num_workers=cores
    )  # type: ignore

    # Define model
    model = LayoutXLMModule(initial_lr=float(args.lr))

    checkpoint_callback = ModelCheckpoint(
        monitor='validation_loss',
        save_top_k=1,
        # save_last=True,
        mode="min")
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        mode="min",
        patience=50
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    mlflow.set_tracking_uri(args.remote_server_uri)
    print(f'Experiment name: {args.experiment_name}')
    mlflow.set_experiment(args.experiment_name)
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name=args.run_name):
        trainer = pl.Trainer(
            accelerator="auto",
            strategy="auto",
            callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
            max_epochs=300,
            accumulate_grad_batches=1,
            num_sanity_val_steps=0,
        )

        trainer.fit(model, datamodule=data_module)
        trainer.test(datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--remote-server-uri',
        default='https://mlflow.lab.sspcloud.fr'
    )
    parser.add_argument('--experiment-name', default='ticket_extraction')
    parser.add_argument('--run-name', default='default')
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--s3', dest='s3', action='store_true')
    parser.add_argument('--lr', default=0.0004)
    parser.add_argument('--batch-size', default=8)
    parser.set_defaults(s3=False)

    args = parser.parse_args()
    main(args)
