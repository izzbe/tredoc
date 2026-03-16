import wandb
from dataclasses import asdict

def init_run(config):
    if wandb.run: return wandb.run;

    run = wandb.init(
        entity="ianzhang-tredoc",
        project="tredoc",
        config=asdict(config)
    )

    return run
