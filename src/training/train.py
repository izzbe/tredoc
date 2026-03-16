from dev_util.logging import get_logger, setup_logger
from dataclasses import dataclass, asdict
from unsloth import FastLanguageModel
from typing import Protocol
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

setup_logger()

logger = get_logger(
    __name__,
    file_handler=True,
    logging_level='info'
)

@dataclass
class ModelArgs:
    name: str
    model_name: str
    max_seq_length: int
    load_in_4bit: bool

@dataclass
class PEFTArgs:
    r: int
    lora_alpha: int
    lora_dropout: float
    random_state: int

    # OPTIONAL CONFIGS
    target_modules: list[str] = None
    bias: str = 'none'
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    loftq_config = None

@dataclass
class TokenizerArgs:
     chat_template: str

@dataclass
class FineTuneRunConfig:
    model_args: ModelArgs
    peft_args: PEFTArgs
    tokenizer_args: TokenizerArgs


class DataFormatter(Protocol):
    def format_dataset_train(self, dataset: Dataset, tokenizer): ...
    def format_dataset_run(self, dataset: Dataset | list[dict[str, str]], tokenizer): ...

class ModelFineTuner:
    def __init__(self, config: FineTuneRunConfig, data_formatter: DataFormatter):
        model, tokenizer = FastLanguageModel.from_pretrained(
            **asdict(config.model_args)
        )
        model = FastLanguageModel.get_peft_model(
            **asdict(config.peft_args)
        )
        tokenizer = get_chat_template(
            tokenizer,
            **asdict(config.tokenizer_args)
        )
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset
        )

