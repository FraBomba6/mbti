import classifier as classifier
from preprocessing import mbti_dataset, i_e, n_s, t_f, j_p
from custom_tokenize import tokenize
from rich.console import Console
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset
from transformers import get_linear_schedule_with_warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3  # number of iterations we are performing the training steps over the dataset
BATCH_SIZE = 4  # number of samples we are using to update the model's parameters
LR = 5e-3  # learning rate

console = Console()
#%%
model_string = 'roberta'
tokenized_text = tokenize(model_string, mbti_dataset)
console.log("Creating dataset and dataloader")
dataset = torch.utils.data.TensorDataset(tokenized_text['input_ids'], tokenized_text['attention_mask'], i_e, n_s, t_f, j_p)
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)  # takes the dataset and shuffels them totally random into the batches of the size 16
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#%%

# Training part
console.log("Generating models, optimizers and schedulers")
# Model I-E

model_ie = classifier.get_model(model_string)
optimizer_ie = AdamW(model_ie.parameters(), lr=LR, eps=1e-8)
scheduler_ie = get_linear_schedule_with_warmup(optimizer_ie, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))

# Model N-S

model_ns = classifier.get_model(model_string)
optimizer_ns = AdamW(model_ns.parameters(), lr=LR, eps=1e-8)
scheduler_ns = get_linear_schedule_with_warmup(optimizer_ns, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))

# Model T-F

model_tf = classifier.get_model(model_string)
optimizer_tf = AdamW(model_tf.parameters(), lr=LR, eps=1e-8)
scheduler_tf = get_linear_schedule_with_warmup(optimizer_tf, num_warmup_steps=0, num_training_steps=EPOCHS*len(train_dataloader))

# Model J-P

model_jp = classifier.get_model(model_string)
optimizer_jp = AdamW(model_jp.parameters(), lr=LR, eps=1e-8)
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
    return avg_loss

# Testing part


def f1(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return f1_score(labels_flat, preds_flat)


def accuracy(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum((preds_flat == labels_flat).numpy())/len(preds_flat)


def test_classification(dataloader, label_index, model):
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

        logits = logits.cpu()
        batch_labels = batch_labels.cpu()

        total_accuracy += accuracy(logits, batch_labels)

    avg_loss = total_loss / len(dataloader)
    console.log("Average loss: {0:.4f}".format(avg_loss))
    avg_accuracy = total_accuracy / len(dataloader)
    console.log("Accuracy: {0:.4f}".format(avg_accuracy))
    model.cpu()
    return avg_accuracy


# running in epochs
test_classification(test_dataloader, 1, model_ie)
test_classification(test_dataloader, 2, model_ns)
test_classification(test_dataloader, 3, model_tf)
test_classification(test_dataloader, 4, model_jp)

best_ie_accuracy = 0
best_ns_accuracy = 0
best_tf_accuracy = 0
best_jp_accuracy = 0
for epoch in range(EPOCHS):
    console.log("\nIE model")
    train_model_one_epoch(train_dataloader, epoch, 1, model_ie, scheduler_ie, optimizer_ie)
    current_ie_accuracy = test_classification(test_dataloader, 1, model_ie)
    if current_ie_accuracy > best_ie_accuracy:
        best_ie_accuracy = current_ie_accuracy
        model_ie.save_pretrained("../models/" + model_string + "_ie")
    console.log("\nNS model")
    train_model_one_epoch(train_dataloader, epoch, 2, model_ns, scheduler_ns, optimizer_ns)
    current_ns_accuracy = test_classification(test_dataloader, 2, model_ns)
    if current_ns_accuracy > best_ns_accuracy:
        best_ns_accuracy = current_ns_accuracy
        model_ns.save_pretrained("../models/" + model_string + "_ns")
    console.log("\nTF model")
    train_model_one_epoch(train_dataloader, epoch, 3, model_tf, scheduler_tf, optimizer_tf)
    current_tf_accuracy = test_classification(test_dataloader, 3, model_tf)
    if current_tf_accuracy > best_tf_accuracy:
        best_tf_accuracy = current_tf_accuracy
        model_tf.save_pretrained("../models/" + model_string + "_tf")
    console.log("\nJP model")
    train_model_one_epoch(train_dataloader, epoch, 4, model_jp, scheduler_jp, optimizer_jp)
    current_jp_accuracy = test_classification(test_dataloader, 4, model_jp)
    if current_jp_accuracy > best_jp_accuracy:
        best_jp_accuracy = current_jp_accuracy
        model_jp.save_pretrained("../models/" + model_string + "_jp")
