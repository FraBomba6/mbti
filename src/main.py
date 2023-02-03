import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# %%
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
