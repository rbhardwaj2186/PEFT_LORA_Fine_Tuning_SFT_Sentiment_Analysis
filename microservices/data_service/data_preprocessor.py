# microservices/data_service/data_preprocessor.py

from transformers import AutoTokenizer

def tokenize_imdb_dataset(train_ds, test_ds, tokenizer_name="distilbert-base-uncased", max_length=256):
    """
    Applies a tokenizer to the IMDB train/test datasets.
    Returns tokenized train_ds and test_ds as Pytorch tensors.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Map tokenization over the dataset
    train_ds = train_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.map(tokenize_function, batched=True)

    # Set format to Pytorch
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"]) # attention_mask -> which token to pay attention to.
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"]) # input_ids -> encoded text.; label-> target values

    return train_ds, test_ds
# We are storing the labels under label column automatically for IMDB. For custom datasets, you might rename columns.
