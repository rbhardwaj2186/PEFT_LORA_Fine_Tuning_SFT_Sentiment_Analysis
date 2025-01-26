# microservices/model_service/infer.py

import torch

def infer_text(model, tokenizer, text, device="cuda"):

    """
    Predict sentiment for a single text input.
    Returns label (0 or 1 for IMDB)
    """

    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)

    return pred.item()

