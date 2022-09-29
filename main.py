import argparse
import logging
import os
from datetime import datetime

from pykeen.hpo import hpo_pipeline
from pykeen.pipeline.api import pipeline

ARTIFACTS_PATH = "./artifacts"


def init_logging() -> None:
    logging.basicConfig(
        filename=os.path.join(ARTIFACTS_PATH, "eval.log"),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def train_with_hpo(kgc_model_name: str, dataset_name: str, artifacts_path: str) -> None:
    logging.debug(f'Optimize hyper parameters for {kgc_model_name}')
    hpo_result = hpo_pipeline(
        n_trials=5,
        model=kgc_model_name,
        dataset=dataset_name,
        stopper='early',  # Terminate unpromising trials
        training_kwargs=dict(num_epochs=10),
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project='primKgEval',
            tags=[f'{kgc_model_name}_{dataset_name}_hpo_' + datetime.now().strftime("%d/%m/%Y %H:%M:%S")],
            reinit=True
        ),
    )

    hpo_result_path = os.path.join(artifacts_path, "hpo_result")
    hpo_result.save_to_directory(hpo_result_path)


def train(kgc_model_name: str, dataset_name: str, artifacts_path: str) -> None:
    logging.debug(f'Train {kgc_model_name} model (no HPO)')
    pip_result = pipeline(
        model=kgc_model_name,
        dataset=dataset_name,
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project='primKgEval',
        ),
        metadata=dict(
            title=f'{kgc_model_name}_{dataset_name}_' + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )

    result_path = os.path.join(artifacts_path, "result")
    pip_result.save_to_directory(result_path)


def main():
    if not os.path.exists(ARTIFACTS_PATH):
        os.mkdir(ARTIFACTS_PATH, mode=777)
    init_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--models", type=str, nargs='+')
    parser.add_argument("--hpo", action='store_true', default=False)

    args = parser.parse_args()

    for kgc_model in args.models:
        model_artifacts_path = os.path.join(ARTIFACTS_PATH, kgc_model)
        if args.hpo:
            train_with_hpo(kgc_model, args.dataset, model_artifacts_path)
        else:
            train(kgc_model, args.dataset, model_artifacts_path)


if __name__ == "__main__":
    main()
