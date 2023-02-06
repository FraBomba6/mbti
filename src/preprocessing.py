#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
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

# %% Encode Type letters to 0 and 1
mbti_dataset['I-E'] = mbti_dataset['I-E'].str.replace("I", '0')
mbti_dataset['I-E'] = mbti_dataset['I-E'].str.replace("E", '1')
mbti_dataset['N-S'] = mbti_dataset['N-S'].str.replace("N", '0')
mbti_dataset['N-S'] = mbti_dataset['N-S'].str.replace("S", '1')
mbti_dataset['T-F'] = mbti_dataset['T-F'].str.replace("T", '0')
mbti_dataset['T-F'] = mbti_dataset['T-F'].str.replace("F", '1')
mbti_dataset['J-P'] = mbti_dataset['J-P'].str.replace("J", '0')
mbti_dataset['J-P'] = mbti_dataset['J-P'].str.replace("P", '1')

# %%
i_e = mbti_dataset['I-E'].tolist()
i_e = [int(x) for x in i_e]
#i_e = tf.convert_to_tensor(i_e)
n_s = mbti_dataset['N-S'].tolist()
n_s = [int(x) for x in n_s]
#n_s = tf.convert_to_tensor(n_s)
t_f = mbti_dataset['T-F'].tolist()
t_f = [int(x) for x in t_f]
#t_f = tf.convert_to_tensor(t_f)
j_p = mbti_dataset['J-P'].tolist()
j_p = [int(x) for x in j_p]
#j_p = tf.convert_to_tensor(j_p)

type_data = tf.stack([i_e, n_s, t_f, j_p], axis=1)
#type_data = tf.convert_to_tensor(type_data)
print(type_data)









