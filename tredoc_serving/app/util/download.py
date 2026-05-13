import wandb

api = wandb.Api()
model_artifact = api.artifact("ianzhang-tredoc/tredoc/model:v7")
model_artifact.download(root="../artifacts/model")

tokenizer_artifact = api.artifact("ianzhang-tredoc/tredoc/tokenizer:v2")
tokenizer_artifact.download(root="../artifacts/tokenizer")