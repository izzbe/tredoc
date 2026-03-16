from dataclasses import asdict, dataclass
from typing import Protocol

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import wandb

from dev_util.logging import get_logger, setup_logger

type Message = dict[str, str]
type Conversation = list[Message]

setup_logger()

logger = get_logger(__name__, file_handler=True, logging_level="info")

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
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = False
    loftq_config = None


@dataclass
class TokenizerArgs:
    chat_template: str


@dataclass
class TrainerConfig:
    dataset_text_field: str
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
    report_to = "WandB"


@dataclass
class ResponseFilter:
    instruction_part: str
    response_part: str


@dataclass
class InferenceConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: float


@dataclass
class FineTuneRunConfig:
    model_args: ModelArgs
    peft_args: PEFTArgs
    tokenizer_args: TokenizerArgs
    training_config: TrainerConfig
    response_filter: ResponseFilter
    inference_config: InferenceConfig


class DataLoader(Protocol):
    def load_dataset_train(self) -> Dataset: ...
    def load_dataset_validation(self) -> Dataset: ...
    def format_dataset_training(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset: ...
    def format_dataset_inference(
        self, messages: list[Conversation], tokenizer: PreTrainedTokenizerBase
    ) -> list[str]: ...
    def get_generation_sample(self) -> Conversation | list[Conversation]: ...


class ModelFineTuner:
    def __init__(self, config: FineTuneRunConfig, data_loader: DataLoader, run_logger: wandb.Run) -> None:
        self.config = config
        self.run: wandb.Run = run_logger
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            **asdict(config.model_args)
        )
        self.model = FastLanguageModel.get_peft_model(**asdict(config.peft_args))
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
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=SFTConfig(**asdict(config.training_config)),
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
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to("cuda")
        input_lens = inputs.attention_mask.sum(dim=1)

        result = self.model.generate(**inputs, **asdict(self.config.inference_config))
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
        model_artifact = wandb.Artifact(
            name="model", type="model", metadata=asdict(self.config)
        )
        model_name = self.config.model_args.name.split("/")[1]
        model_artifact.add_dir(f"tredoc-{model_name}-merged")
        self.run.log_artifact(model_artifact)

        train_artifact = wandb.Artifact(
            name="train_data",
            type="dataset",
        )
        train_artifact.add_file("./data/train.jsonl")
        self.run.log_artifact(train_artifact)

        test_artifact = wandb.Artifact(
            name="test_data",
            type="dataset",
        )
        test_artifact.add_file("./data/test.jsonl")
        self.run.log_artifact(test_artifact)

        tokenizer_artifact = wandb.Artifact(name="tokenizer", type="dataset")
        tokenizer_artifact.add_dir(f"tokenizer-{model_name}")
        self.run.log_artifact(tokenizer_artifact)

        sample_result = self.generate(self.data_loader.get_generation_sample())
        table = wandb.Table(columns=sample_result[0].keys())
        for generation in sample_result:
            row = [item for _, item in generation.items()]
            table.add_data(*row)

        self.run.log({"sample_generations": table})
