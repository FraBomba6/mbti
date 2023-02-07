# IF YOU RUN FROM TERMINAL
import classifier as classifier
from preprocessing import mbti_dataset, i_e, n_s, t_f, j_p
from custom_tokenize import tokenize

# IF YOU RUN CHUNKS
# import src.classifier as classifier
# from src.preprocessing import mbti_dataset
# from src.custom_tokenize import tokenize

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3  # number of iterations we are performing the training steps over the dataset
BATCH_SIZE = 16  # number of samples we are using to update the model's parameters

#%%
model_string = 'xlnet'
tokenized_text = tokenize(model_string, mbti_dataset)
dataset = torch.utils.data.TensorDataset(tokenized_text['input_ids'], tokenized_text['attention_mask'], i_e, n_s, t_f, j_p)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
# dataset = torch.cat((type_data, tokenized_text['input_ids']), 1)
print(dataset)

#%%
model = classifier.get_model("xlnet")
model.to(DEVICE)  # send the model to the device for usage

optimizer = AdamW(model.parameters(), lr=5e-3, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))


