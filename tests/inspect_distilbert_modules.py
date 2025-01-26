# inspect_distilbert_modules.py

from transformers import AutoModelForSequenceClassification

def print_module_names(model_name="distilbert-base-uncased"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    for name, module in model.named_modules():
        print(name)

if __name__ == "__main__":
    print_module_names()
