from transformers import AutoTokenizer
import pandas as pd
from rich.console import Console
console = Console()


# %% get tokenizer for corresponding model
def get_tokenizer(model_string: str):
    console.log(f"Getting tokenizer for {model_string} model")
    if model_string == "roberta":
        tokenizer_string = "roberta-base"
    elif model_string == "xlnet":
        tokenizer_string = "xlnet-base-cased"
    return AutoTokenizer.from_pretrained(tokenizer_string)


def tokenize(model_string, df: pd.DataFrame):
    console.log(f"Tokenizing data using {model_string} tokenizer")
    tokenizer = get_tokenizer(model_string)
    text = df['posts'].tolist()
    tokenized_text = tokenizer(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    return tokenized_text




