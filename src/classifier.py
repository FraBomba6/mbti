from transformers import XLNetForSequenceClassification, RobertaForSequenceClassification


def get_model(model_string: str):
    if model_string == "xlnet":
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=2)
    elif model_string == "roberta":
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    else:
        raise Exception("The model is neither XLNet nor RoBERTa. Valid arguments: xlnet, roberta")
    return model
