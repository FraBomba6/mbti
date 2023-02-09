from transformers import XLNetForSequenceClassification, RobertaForSequenceClassification
import torch
from rich.console import Console
rich = Console()


def get_model(model_string: str):
    rich.log(f"Getting model for {model_string} model")
    if model_string == "xlnet":
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
    elif model_string == "roberta":
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion", num_labels=2)
        model.classifier = torch.nn.Linear(768, 2)
    else:
        raise Exception("The model is neither XLNet nor RoBERTa. Valid arguments: xlnet, roberta")
    return model
