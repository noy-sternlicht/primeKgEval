import argparse
import logging
import configparser
import os
import shutil

import pystow

from config import CONFIG_PATH
from datetime import datetime

from pykeen.hpo import hpo_pipeline
from pykeen.pipeline.api import pipeline

daytime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
config = configparser.ConfigParser()
config.read(CONFIG_PATH)


def init_logging(dataset_name: str) -> str:
    art_path = os.path.join('..', config['DEFAULT']['artifact_dir'])
    if not os.path.exists(art_path):
        os.mkdir(art_path, mode=0o777)

    run_path = os.path.join(art_path, f'Run_{dataset_name}_{daytime}')
    os.mkdir(run_path, mode=0o777)

    logging.basicConfig(
        filename=os.path.join(run_path, "eval.log"),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

    return run_path


def train_with_hpo(kgc_model_name: str, dataset_name: str, artifacts_path: str) -> None:
    logging.debug(f'Optimize hyper parameters for {kgc_model_name}')
    hpo_result = hpo_pipeline(
        n_trials=config['HPO'].getint('n_trials'),
        model=kgc_model_name,
        dataset=dataset_name,
        sampler=config['HPO']['sampler'],
        training_kwargs=dict(num_epochs=config['TRAINING'].getint('n_epochs')),
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project=config['WANDB']['project_name'],
            tags=[f'{kgc_model_name}_{dataset_name}_hpo_' + daytime],
            reinit=True
        ),
    )

    hpo_result_path = os.path.join(artifacts_path, config['HPO']['result_dir'])
    hpo_result.save_to_directory(hpo_result_path)


def train(kgc_model_name: str, dataset_name: str, artifacts_path: str) -> None:
    logging.debug(f'Train {kgc_model_name} model (no HPO)')
    pip_result = pipeline(
        model=kgc_model_name,
        dataset=dataset_name,
        training_kwargs=dict(num_epochs=config['TRAINING'].getint('n_epochs')),
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project=config['WANDB']['project_name'],
        ),
        metadata=dict(
            title=f'{kgc_model_name}_{dataset_name}_' + daytime
        )
    )

    result_path = os.path.join(artifacts_path, "result")
    pip_result.save_to_directory(result_path)


def drain_pykeen_artifacts_to_model_dir(model_artifacts_path: str, dataset_name: str) -> None:
    pykeen_artifacts_path = os.path.join(model_artifacts_path, "pykeen_artifacts")
    os.mkdir(pykeen_artifacts_path, mode=0o777)
    pykeen_datasets_dir_path = pystow.join('pykeen', 'datasets')
    shutil.move(os.path.join(pykeen_datasets_dir_path, dataset_name.lower()), pykeen_artifacts_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--models", type=str, nargs='+', help="A list of KGE models to evaluate")
    parser.add_argument("--hpo", action='store_true', default=False, help="Preforms hyperparameter optimization if set")

    args = parser.parse_args()

    run_logs_path = init_logging(args.dataset)

    for kgc_model in args.models:
        model_artifacts_path = os.path.join(run_logs_path, kgc_model)
        if args.hpo:
            train_with_hpo(kgc_model, args.dataset, model_artifacts_path)
        else:
            train(kgc_model, args.dataset, model_artifacts_path)

        drain_pykeen_artifacts_to_model_dir(model_artifacts_path, args.dataset)


if __name__ == "__main__":
    main()
