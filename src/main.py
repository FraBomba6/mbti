# IF YOU RUN FROM TERMINAL
import classifier as classifier
from preprocessing import mbti_dataset, type_data
from custom_tokenize import tokenize
import tensorflow as tf

# IF YOU RUN CHUNKS
#import src.classifier as classifier
#from src.preprocessing import mbti_dataset
#from src.custom_tokenize import tokenize

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3 # number of iterations we are performing the training steps over the dataset

#%%
model_string = 'xlnet'
tokenized_text = tokenize(model_string, mbti_dataset)
#type_data = type_data
dataset = tf.concat([type_data], tokenized_text['input_ids'])
print()

#%%
model = classifier.get_model("xlnet")
model.to(DEVICE) # send the model to the device for usage

optimizer = AdamW(model.parameters(), lr=5e-3, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))


