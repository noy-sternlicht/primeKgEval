import argparse
import logging

from pykeen.pipeline import pipeline, PipelineResult


def init_logging() -> None:
    logging.basicConfig(
        filename="eval.log",
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def get_relevant_metrics(pipe_result: PipelineResult) -> str:
    hits1 = pipe_result.get_metric("hits_at_1")
    hits3 = pipe_result.get_metric("hits_at_3")
    hits10 = pipe_result.get_metric("hits_at_10")
    mrr = pipe_result.get_metric("inverse_harmonic_mean_rank")

    return f'H@1: {hits1} H@3: {hits3}, H@10: {hits10}, MRR: {mrr}'


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--models", type=str, nargs='+')

    args = parser.parse_args()

    for kgc_model in args.models:
        result = pipeline(
            model=kgc_model,
            dataset=args.dataset,
            device="cpu"
        )

        eval_metrics = get_relevant_metrics(result)
        logging.info(f'[Model={kgc_model}] {eval_metrics}')
