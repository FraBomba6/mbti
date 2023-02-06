from transformers import XLNetForSequenceClassification, RobertaForSequenceClassification
from preprocessing import mbti_dataset

def get_model(model_string: str):
    if model_string == "xlnet":
        model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
    elif model_string == "roberta":
        model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
    else:
        raise Exception("The model is neither XLNet nor RoBERTa. Valid arguments: xlnet, roberta")
    return model

#model_e_i = get_model(model_string)
label_e_i = mbti_dataset['I-E']
label_n_s = mbti_dataset['N-S']
label_t_f = mbti_dataset['T-F']
label_j_p = mbti_dataset['J-P']

# Classifying

