# microservices/orchestrator/config.py

# SFT Configs
DATASET_NAME = "imdb"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
DEVICE = "cuda"

# RLHF Configs
PEFT_CONFIG = {
   "quantization": {
       "load_in_4bit": True,
       "compute_dtype": "float16",
       "quant_type": "nf4"
   },
   "lora": {
       "r": 16,
       "alpha": 32,
       "dropout": 0.05,
       "target_modules": ["q_proj", "v_proj"]
   }
}

PPO_CONFIG = {
   "batch_size": 8,
   "learning_rate": 1.41e-5,
   "max_epochs": 1
}