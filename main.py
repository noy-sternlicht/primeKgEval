import argparse
import logging
import configparser
import os
from config import CONFIG_PATH
from datetime import datetime

from pykeen.hpo import hpo_pipeline
from pykeen.pipeline.api import pipeline

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


def init_logging() -> None:
    logging.basicConfig(
        filename=os.path.join(config['DEFAULT']['artifact_dir'], "eval.log"),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def train_with_hpo(kgc_model_name: str, dataset_name: str, artifacts_path: str) -> None:
    logging.debug(f'Optimize hyper parameters for {kgc_model_name}')
    hpo_result = hpo_pipeline(
        n_trials=config['HPO'].getint('n_trials'),
        model=kgc_model_name,
        dataset=dataset_name,
        sampler=config['HPO']['sampler'],
        stopper=config['HPO']['stopper'],
        training_kwargs=dict(num_epochs=config['TRAINING'].getint('n_epochs')),
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project=config['WANDB']['project_name'],
            tags=[f'{kgc_model_name}_{dataset_name}_hpo_' + datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
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
            title=f'{kgc_model_name}_{dataset_name}_' + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )

    result_path = os.path.join(artifacts_path, "result")
    pip_result.save_to_directory(result_path)


def main():
    artifact_path = os.path.join("./", config['DEFAULT']['artifact_dir'])

    if not os.path.exists(artifact_path):
        os.mkdir(artifact_path, mode=777)
    init_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--models", type=str, nargs='+')
    parser.add_argument("--hpo", action='store_true', default=False)

    args = parser.parse_args()

    for kgc_model in args.models:
        model_artifacts_path = os.path.join(artifact_path, kgc_model)
        if args.hpo:
            train_with_hpo(kgc_model, args.dataset, model_artifacts_path)
        else:
            train(kgc_model, args.dataset, model_artifacts_path)


if __name__ == "__main__":
    main()
