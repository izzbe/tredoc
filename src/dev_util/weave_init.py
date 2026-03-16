from dataclasses import asdict

import wandb

from training.train import FineTuneRunConfig


def init_run(config: FineTuneRunConfig) -> wandb.Run:
    if wandb.run:
        return wandb.run
    model_name = config.model_args.name.split("/")[1]
    run = wandb.init(
        entity="ianzhang-tredoc",
        project="tredoc",
        name=model_name,
        config=asdict(config),
    )

    return run
