#importing XLNet
from transformers import XLNetTokenizer, XLNetModel
from simpletransformers.classification import ClassificationModel
import pandas as pd
import logging
import sklearn
import torch.nn as nn
import preprocessing as prep
import pandas as pd
import re
import os
import math
import torch
# import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, NLLLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
from transformers import AdamW, XLNetTokenizer, XLNetModel, TFXLNetModel, XLNetLMHeadModel, XLNetConfig, XLNetForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import unicodedata
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,roc_auc_score
import seaborn as sns
import itertools
import plotly
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter


#df = pd.read_csv('dataset.csv')
#train_df=pd.DataFrame('dataset.csv')
#test_df=pd.DataFrame('dataset.csv')

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state


# followed instructions of "XLNet Fine-Tuning Tutorial with PyTorch" https://mccormickml.com/2019/09/19/XLNet-fine-tuning/
import torch
import sentencepiece as sp
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# %%
#setting hyperparameters
train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 1,
    'learning_rate': 0.00001,
    'weight_decay': 0.01,
    'train_batch_size': 128,
    'fp16': True,
    'output_dir': '/outputs/',
}

#%%
#Training the model


logging.basicConfig(level=logging.DEBUG)
transformers_logger = logging.getLogger('transformers')
transformers_logger.setLevel(logging.WARNING)

# We use the XLNet base cased pre-trained model.
#model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=2, args=train_args)

# Train the model, there is no development or validation set for this dataset
# https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
#model.train_model(train_df)

# Evaluate the model in terms of accuracy score
#result, model_outputs, wrong_predictions = model.eval_model(test_df, acc=sklearn.metrics.accuracy_score)

class XLNet_Model(nn.Module):
  def __init__(self, classes):
    super(XLNet_Model, self).__init__()
    self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
    self.out = nn.Linear(self.xlnet.config.hidden_size, classes)
  def forward(self, input):
    outputs = self.xlnet(**input)
    out = self.out(outputs.last_hidden_state)
    return out