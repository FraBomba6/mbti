import pandas as pd
import torch
import numpy as np
from rich.console import Console
console = Console()

# %%
console.log("Reading dataset")
mbti_dataset = pd.read_csv("../data/dataset.csv")

# %% LOWERCASE
console.log("Lowercasing")
mbti_dataset['posts'] = mbti_dataset['posts'].str.lower()

# %% REMOVE HYPERLINKS, SPECIAL, CHARACTERS, AND MBTI TYPES -> STOPWORDS WILL BE REMOVED AND LEMMATIZATION WILL BE DONE WITH THE TOKENIZER
console.log("Removing hyperlinks, special characters, and MBTI types")
words_to_remove = [r'http\S+', '|', '_', 'infp', 'infj', 'intp', 'intj', 'entp', 'enfp', 'istp', 'isfp', 'entj', 'istj', 'enfj', 'isfj', 'estp', 'esfp', 'esfj', 'estj']
for word in words_to_remove:
    mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(word, '', regex=True)

# %%
console.log("Removing punctuation")
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r'[^\w\s\']', ' ', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r'\s{2,}', ' ', regex=True)


# %% CONVERT SMILEY FACES INTO WORDS
console.log("Converting smiley faces into words")
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\)|:\)", 'happy', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\(|:\(", 'sad', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r";-\)|;\)", 'wink', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-D|:D", 'grin', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-O|:o|:-o|:O", 'surprised', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-P|:P", 'tongue_out', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\||:\|", 'blank', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-\|/:\/", 'disappointed', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r":-S|:S", 'confused', regex=True)

# %% Split the type column into four columns and store the result in a temporary DataFrame
console.log("Splitting type column into four columns")
temp_df = mbti_dataset['type'].str.split("", expand=True)

# Assign the columns of the temporary DataFrame to the original DataFrame
mbti_dataset['I-E'] = temp_df[1]
mbti_dataset['N-S'] = temp_df[2]
mbti_dataset['T-F'] = temp_df[3]
mbti_dataset['J-P'] = temp_df[4]

# Specify the new order of columns
new_column_order = ['type', 'I-E', 'N-S', 'T-F', 'J-P', 'posts']

# Reorder the columns in the DataFrame
mbti_dataset = mbti_dataset.reindex(columns=new_column_order)

# %% Encode Type letters to 0 and 1
console.log("Encoding type letters to 0 and 1")
mbti_dataset['I-E'] = mbti_dataset['I-E'].apply(lambda x: torch.from_numpy(np.array([1.0, 0.0])) if x == 'I' else torch.from_numpy(np.array([0.0, 1.0])))
mbti_dataset['N-S'] = mbti_dataset['N-S'].apply(lambda x: torch.from_numpy(np.array([1.0, 0.0])) if x == 'N' else torch.from_numpy(np.array([0.0, 1.0])))
mbti_dataset['T-F'] = mbti_dataset['T-F'].apply(lambda x: torch.from_numpy(np.array([1.0, 0.0])) if x == 'T' else torch.from_numpy(np.array([0.0, 1.0])))
mbti_dataset['J-P'] = mbti_dataset['J-P'].apply(lambda x: torch.from_numpy(np.array([1.0, 0.0])) if x == 'J' else torch.from_numpy(np.array([0.0, 1.0])))

# %%
i_e = mbti_dataset['I-E'].tolist()
i_e = torch.stack(i_e)
n_s = mbti_dataset['N-S'].tolist()
n_s = torch.stack(n_s)
t_f = mbti_dataset['T-F'].tolist()
t_f = torch.stack(t_f)
j_p = mbti_dataset['J-P'].tolist()
j_p = torch.stack(j_p)
