from dataclasses import asdict
from datetime import datetime

import wandb

from training.train import FineTuneRunConfig


def init_run(config: FineTuneRunConfig) -> wandb.Run:
    if wandb.run:
        return wandb.run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_args.model_name.split("/")[1]
    run = wandb.init(
        entity="ianzhang-tredoc",
        project="tredoc",
        name=(model_name+'_'+timestamp),
        config=asdict(config),
    )

    return run
