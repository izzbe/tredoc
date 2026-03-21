from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from dataclasses import asdict, dataclass
from typing import Protocol

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from dev_util.dir import OUTPUT
import wandb
import yaml

from dev_util.logging import get_logger, setup_logger

type Message = dict[str, str]
type Conversation = list[Message]

setup_logger()

logger = get_logger(__name__, file_handler=True, logging_level="info")

@dataclass
class ModelArgs:
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
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    loftq_config = None


@dataclass
class TokenizerArgs:
    chat_template: str


@dataclass
class TrainerConfig:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    num_train_epochs: int
    max_steps: int
    learning_rate: float
    logging_steps: int
    optim: str
    weight_decay: float
    lr_scheduler_type: str
    seed: int
    report_to: str = "wandb"
    eval_strategy: str = "steps"
    eval_steps: int = 5

@dataclass
class ResponseFilter:
    instruction_part: str
    response_part: str


@dataclass
class InferenceConfig:
    max_new_tokens: int
    temperature: float
    top_p: float = 1
    top_k: int = 0
    min_p: float = 0
    do_sample: bool = False


@dataclass
class FineTuneRunConfig:
    model_args: ModelArgs
    peft_args: PEFTArgs
    tokenizer_args: TokenizerArgs
    trainer_config: TrainerConfig
    response_filter: ResponseFilter
    inference_config: InferenceConfig

def get_config(config_path: str) -> FineTuneRunConfig:
    with open(config_path, mode='r') as f:
        config_args = yaml.safe_load(f)
    return FineTuneRunConfig(
        model_args=ModelArgs(**config_args['model_args']),
        peft_args=PEFTArgs(**config_args['peft_args']),
        tokenizer_args=TokenizerArgs(**config_args['tokenizer_args']),
        trainer_config=TrainerConfig(**config_args['trainer_config']),
        response_filter=ResponseFilter(**config_args['response_filter']),
        inference_config=InferenceConfig(**config_args['inference_config'])
    )

class DataLoader(Protocol):
    def load_dataset_train(self) -> Dataset: ...
    def load_dataset_validation(self) -> Dataset: ...
    def format_dataset_training(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset: ...
    def format_dataset_inference(
        self, messages: list[Conversation], tokenizer: PreTrainedTokenizerBase
    ): ...
    def get_generation_sample(self) -> Conversation | list[Conversation]: ...


class ModelFineTuner:
    def __init__(self, config: FineTuneRunConfig, data_loader: DataLoader, run_logger: wandb.Run) -> None:
        self.config = config
        self.run: wandb.Run = run_logger
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            **asdict(config.model_args)
        )
        self.model = FastLanguageModel.get_peft_model(self.model, **asdict(config.peft_args))
        self.tokenizer = get_chat_template(
            self.tokenizer, **asdict(config.tokenizer_args)
        )

        self.train_dataset = data_loader.load_dataset_train()
        self.train_dataset = data_loader.format_dataset_training(
            self.train_dataset, self.tokenizer
        )

        self.eval_dataset = data_loader.load_dataset_validation()
        self.eval_dataset = data_loader.format_dataset_training(
            self.eval_dataset, self.tokenizer
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer),
            args=SFTConfig(dataset_text_field="text",
                           **asdict(config.trainer_config)),
        )

        self.trainer = train_on_responses_only(
            self.trainer, **asdict(config.response_filter)
        )
        self.train_stats = None
        self.data_loader = data_loader

    def train(self):
        self.train_stats = self.trainer.train()
        return self.train_stats

    def generate(
        self, messages: Conversation | list[Conversation]
    ) -> list[dict[str, str]]:
        if isinstance(messages[0], dict):
            messages = [messages]

        text = self.data_loader.format_dataset_inference(messages, self.tokenizer)
        input_lens = text.attention_mask.sum(dim=1)

        result = self.model.generate(**text, use_cache=False, **asdict(self.config.inference_config))
        return [
            {
                "input": self.tokenizer.decode(
                    seq[: input_lens[i]], skip_special_tokens=True
                ),
                "output": self.tokenizer.decode(
                    seq[input_lens[i] :], skip_special_tokens=True
                ),
            }
            for i, seq in enumerate(result)
        ]

    def save(self):
        self.run = wandb.init(project="tredoc", id=self.run.id, resume="must")
        model_dir = OUTPUT / "final_model"
        self.model.save_pretrained(model_dir)
        model_artifact = wandb.Artifact(
            name="model", type="model", metadata=asdict(self.config)
        )
        model_artifact.add_dir(model_dir)
        self.run.log_artifact(model_artifact)

        train_data_dir = OUTPUT / "train_data"
        self.train_dataset.save_to_disk(train_data_dir)
        train_artifact = wandb.Artifact(
            name="train_data",
            type="dataset",
        )
        train_artifact.add_dir(train_data_dir)
        self.run.log_artifact(train_artifact)


        eval_data_dir = OUTPUT / "eval_data"
        self.eval_dataset.save_to_disk(eval_data_dir)
        eval_artifact = wandb.Artifact(
            name="eval_data",
            type="dataset",
        )
        eval_artifact.add_dir(eval_data_dir)
        self.run.log_artifact(eval_artifact)

        tok_dir = OUTPUT / "tokenizer"
        self.tokenizer.save_pretrained(tok_dir)
        tokenizer_artifact = wandb.Artifact(name="tokenizer", type="dataset")
        tokenizer_artifact.add_dir(tok_dir)
        self.run.log_artifact(tokenizer_artifact)

        sample_result = self.generate(self.data_loader.get_generation_sample())
        table = wandb.Table(columns=list(sample_result[0].keys()))
        for generation in sample_result:
            row = [item for _, item in generation.items()]
            table.add_data(*row)

        self.run.log({"sample_generations": table})
        self.run.finish()
