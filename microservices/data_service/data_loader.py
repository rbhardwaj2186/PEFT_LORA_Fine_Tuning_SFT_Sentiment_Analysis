# microservices/data_service/data_loader.py

from datasets import load_dataset

def load_imdb_dataset():
    """
    Loads the IMDB dataset from hugging Face
    Returns train and test splits as huggingface Dataset objects
    """

    imdb = load_dataset('imdb')
    return imdb["train"], imdb["test"]