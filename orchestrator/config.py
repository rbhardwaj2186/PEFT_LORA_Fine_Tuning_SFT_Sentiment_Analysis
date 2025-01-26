# microservices/orchestrator/config.py

DATASET_NAME = "imdb"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2

# Hyperparams
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 256

DEVICE = "cuda" # or "cpu"