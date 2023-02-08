# IF YOU RUN FROM TERMINAL
import classifier as classifier
from preprocessing import mbti_dataset, i_e, n_s, t_f, j_p
from custom_tokenize import tokenize
from rich.console import Console
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

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
console = Console()
#%%
model_string = 'xlnet'
tokenized_text = tokenize(model_string, mbti_dataset)
dataset = torch.utils.data.TensorDataset(tokenized_text['input_ids'], tokenized_text['attention_mask'], i_e, n_s, t_f, j_p)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) #takes the dataset and shuffels them totally random into the batches of the size 16
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#%%

# Training part

# Model I-E

model_ie = classifier.get_model("xlnet")
optimizer_ie = AdamW(model_ie.parameters(), lr=5e-3, eps=1e-8)
scheduler_ie = get_linear_schedule_with_warmup(optimizer_ie, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))

# Model N-S

model_ns = classifier.get_model("xlnet")
optimizer_ns = AdamW(model_ns.parameters(), lr=5e-3, eps=1e-8)
scheduler_ns = get_linear_schedule_with_warmup(optimizer_ns, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))

# Model T-F

model_tf = classifier.get_model("xlnet")
optimizer_tf = AdamW(model_tf.parameters(), lr=5e-3, eps=1e-8)
scheduler_tf = get_linear_schedule_with_warmup(optimizer_tf, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))

# Model J-P

model_jp = classifier.get_model("xlnet")
optimizer_jp = AdamW(model_jp.parameters(), lr=5e-3, eps=1e-8)
scheduler_jp = get_linear_schedule_with_warmup(optimizer_jp, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))


def train_model_one_epoch(dataloader, epoch, label_index, model, scheduler, optimizer):
    console.log(f"Training epoch #{epoch+1}")
    total_loss = 0
    model.to(DEVICE)
    model.train()

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_input_ids = batch[0].to(DEVICE)
        batch_input_masks = batch[1].to(DEVICE)
        batch_labels = batch[label_index+1].to(DEVICE)

        model.zero_grad()
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_input_masks,
            labels=batch_labels
        )
        loss, logits = outputs[:2]  # logit = predicted value (that one gets), label is what I want
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent gradient from exploding (normalization)
        optimizer.step()  # updating the values for next iteration
        scheduler.step()  # updating the values for next iteration

    avg_loss = total_loss / len(dataloader)
    console.log("Average loss: {0:.4f}".format(avg_loss))
    model.cpu()

# Testing part

def f1(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return f1_score(labels_flat, preds_flat)

def test_classification(dataloader, label_index, model, scheduler, optimizer):
    console.log(f"Testing")
    total_loss = 0
    total_accuracy = 0
    model.to(DEVICE)
    model.eval()

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_input_ids = batch[0].to(DEVICE)
        batch_input_masks = batch[1].to(DEVICE)
        batch_labels = batch[label_index+1].to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_input_masks,
                labels=batch_labels
            )
            loss, logits = outputs[:2]
            total_loss += loss.item()

        logits.cpu()
        batch_labels.cpu()

        total_accuracy += f1(logits, batch_labels)

    avg_loss = total_loss / len(dataloader)
    console.log("Average loss: {0:.4f}".format(avg_loss))
    avg_accuracy = total_accuracy / len(dataloader)
    console.log("Accuracy: {0:.4f}".format(avg_accuracy))
    model.cpu()


# running in epochs

for epoch in range(EPOCHS):
    train_model_one_epoch(train_dataloader, epoch, 1, model_ie, scheduler_ie, optimizer_ie)
    test_classification(test_dataloader, 1, model_ie, scheduler_ie, optimizer_ie)
    train_model_one_epoch(train_dataloader, epoch, 2, model_ns, scheduler_ns, optimizer_ns)
    test_classification(test_dataloader, 2, model_ns, scheduler_ns, optimizer_ns)
    train_model_one_epoch(train_dataloader, epoch, 3, model_tf, scheduler_tf, optimizer_tf)
    test_classification(test_dataloader, 3, model_tf, scheduler_tf, optimizer_tf)
    train_model_one_epoch(train_dataloader, epoch, 4, model_jp, scheduler_jp, optimizer_jp)
    test_classification(test_dataloader, 4, model_jp, scheduler_jp, optimizer_jp)
