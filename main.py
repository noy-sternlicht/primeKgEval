import argparse
import logging
import os

from pykeen.hpo import hpo_pipeline
from pykeen.pipeline.api import pipeline_from_path, PipelineResult

ARTIFACTS_PATH = "./artifacts"


def init_logging() -> None:
    logging.basicConfig(
        filename=os.path.join(ARTIFACTS_PATH, "eval.log"),
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def get_relevant_metrics(pipe_result: PipelineResult) -> str:
    hits1 = pipe_result.get_metric("hits_at_1")
    hits3 = pipe_result.get_metric("hits_at_3")
    hits10 = pipe_result.get_metric("hits_at_10")
    mrr = pipe_result.get_metric("inverse_harmonic_mean_rank")

    return f'H@1: {hits1} H@3: {hits3}, H@10: {hits10}, MRR: {mrr}'


if __name__ == "__main__":
    if not os.path.exists(ARTIFACTS_PATH):
        os.mkdir(ARTIFACTS_PATH, mode=777)
    init_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--models", type=str, nargs='+')

    args = parser.parse_args()

    for kgc_model in args.models:
        logging.debug(f'Optimize hyper parameters for {kgc_model}')
        hpo_result = hpo_pipeline(
            n_trials=5,
            model=kgc_model,
            dataset=args.dataset,
            device="cpu",
            n_jobs=-1,  # Use all available CPUs
            stopper='early',  # Terminate unpromising trials
            training_kwargs=dict(num_epochs=10)
        )

        model_artifacts_path = os.path.join(ARTIFACTS_PATH, kgc_model)
        hpo_result_path = os.path.join(model_artifacts_path, "hpo_result")
        hpo_result.save_to_directory(hpo_result_path)

        logging.debug(f'Train optimized {kgc_model} model')
        pip_result = pipeline_from_path(os.path.join(hpo_result_path, "best_pipeline/pipeline_config.json"))
        pip_result.save_to_directory(os.path.join(model_artifacts_path, "optimal_pipline_result"))

        eval_metrics = get_relevant_metrics(pip_result)
        logging.info(f'[Model={kgc_model}] {eval_metrics}')
