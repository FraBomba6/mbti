#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from transformers import BertForSequenceClassification, BertTokenizer


# %%
# IF YOU RUN USING CHUNKS
#mbti_dataset = pd.read_csv("data/dataset.csv")
# ELSE
mbti_dataset = pd.read_csv("../data/dataset.csv")

#TODO: decide how to hand the words/sentences to the model: 1 sentence, 2 sentences, half-half,
# tokenization: tokener will split sentence

# %% LOWERCASE
mbti_dataset['posts'] = mbti_dataset['posts'].str.lower()

# %% REMOVE HYPERLINKS, SPECIAL, CHARACTERS, AND MBTI TYPES -> STOPWORDS WILL BE REMOVED AND LEMMATIZATION WILL BE DONE WITH THE TOKENIZER
words_to_remove = [r'http\S+', '|', '_', 'infp', 'infj', 'intp', 'intj', 'entp', 'enfp', 'istp', 'isfp', 'entj', 'istj', 'enfj', 'isfj', 'estp', 'esfp', 'esfj', 'estj']
for word in words_to_remove:
    mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(word, '', regex=True)

# %%
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r'[^\w\s\']', ' ', regex=True)
mbti_dataset['posts'] = mbti_dataset['posts'].str.replace(r'\s{2,}', ' ', regex=True)


# %% CONVERT SMILEY FACES INTO WORDS
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









