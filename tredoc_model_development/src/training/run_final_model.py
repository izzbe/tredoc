from unsloth.chat_templates import standardize_sharegpt
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from dev_util.weave_init import init_run
import training.train as tr

import psycopg2 as psql
import os
from dotenv import load_dotenv
from dev_util.dir import ROOT
import pandas as pd
import re

class TredocDataLoader:
    def __init__(self):
        load_dotenv(ROOT / '.env')
        db = os.environ.get('DB_NAME')
        user = os.environ.get('DB_USER')
        password = os.environ.get('DB_PASSWORD')

        conn = psql.connect(
            host="localhost",
            dbname=db,
            user=user,
            password=password
        )

        cur = conn.cursor()
        cur.execute("SELECT * FROM pairs;")
        data = cur.fetchall()
        self.raw_data = pd.DataFrame(
            data,
            columns=['id', 'repo_id', 'file_path', 'func_name', 'signature', 'body', 'docstring', 'style']
        )

        def remove_docstring(func_text: str) -> str:
            return re.sub(
                r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\s*\n(\s+))'
                r'(?:\2(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'))\n',
                r'\1',
                func_text
            )

        self.raw_data['body'] = self.raw_data['body'].map(remove_docstring)

        def create_conversation(row):
            return [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user",
                 "content": f"Create a docstring for python code following the specifications: <style>{row.style}</style>\n<signature>{row.signature}</signature>\n<body>{row.body}</body>"},
                {"role": "assistant",
                 "content": row.docstring}
            ]

        self.raw_data['conversation'] = self.raw_data.apply(create_conversation, axis=1)

        self.train_data = self.raw_data.sample(frac=0.999, random_state=6876)
        self.test_data = self.raw_data.drop(self.train_data.index)

    def load_dataset_train(self) -> Dataset:
        return Dataset.from_pandas(self.train_data)

    def load_dataset_validation(self) -> Dataset:
        return Dataset.from_pandas(self.test_data)

    def format_dataset_training(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> Dataset:
        def formatting_prompts_func(examples):
            convos = examples["conversation"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                     convos]
            return {"text": texts, }

        return dataset.map(formatting_prompts_func, batched=True)


    def format_dataset_inference(
        self, messages: list[tr.Conversation], tokenizer: PreTrainedTokenizerBase
    ):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        return inputs

    def get_generation_sample(self) -> tr.Conversation | list[tr.Conversation]:
        sample_message = """Create a docstring for python code following the specifications: <style>plain</style>
        <signature>    def generate(
        self, messages: Conversation | list[Conversation]
    )</signature>
        <body>    def generate(
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
        ]</body>"""

        messages = [
            {"role": "user", "content": sample_message}
        ]
        return messages

def main():
    loader = TredocDataLoader()
    config = tr.get_config('final_model_config.yml')
    run = init_run(config)
    refit = tr.ModelFineTuner(config, loader, run)
    refit.train()
    refit.save()

if __name__ == "__main__":
    main()
