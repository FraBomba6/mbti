from transformers import AutoTokenizer
import pandas as pd
#import preprocessing

# %% get tokenizer for corresponding model
def get_tokenizer(model_string: str):
    if model_string == "roberta":
        tokenizer_string = "roberta-base"
    elif model_string == "xlnet":
        tokenizer_string = "xlnet-base-cased"
    return AutoTokenizer.from_pretrained(tokenizer_string)

def tokenize(model_string, df: pd.DataFrame):
    tokenizer = get_tokenizer(model_string)
    text = df['posts'].tolist()
    tokenized_text = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, return_tensors='pt')
    #print(input_ids)
    return tokenized_text




