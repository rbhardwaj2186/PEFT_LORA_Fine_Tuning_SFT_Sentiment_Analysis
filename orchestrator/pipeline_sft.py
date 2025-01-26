# microservices/orchestrator/pipeline_sft.py

import torch
from microservices.data_service.data_loader import load_imdb_dataset
from microservices.data_service.data_preprocessor import tokenize_imdb_dataset
from microservices.model_services.model_definition import create_sft_model
from microservices.model_services.train_sft import train_sft, evaluate_sft
import orchestrator.config as cfg
from torch.utils.data import DataLoader

def run_sft_pipeline(resume=False, resume_checkpoint=None, use_peft=False):
    # 1. Load raw data
    train_ds, test_ds = load_imdb_dataset()
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # 2. Tokenize
    train_ds, test_ds = tokenize_imdb_dataset(
        train_ds,
        test_ds,
        tokenizer_name=cfg.MODEL_NAME,
        max_length=cfg.MAX_LENGTH
    )

    # 3. Create model
    model = create_sft_model(
        model_name=cfg.MODEL_NAME,
        num_labels=cfg.NUM_LABELS,
        use_peft=use_peft  # Pass the flag dynamically
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. Train with optional checkpointing
    trained_model = train_sft(
        model=model,
        train_ds=train_ds,
        test_ds=test_ds,
        device=str(device),
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        lr=cfg.LEARNING_RATE,
        checkpoint_dir="./checkpoints",    # or None if you don't want checkpointing
        resume=resume,
        resume_checkpoint=resume_checkpoint,
        use_peft=use_peft  # Pass the flag dynamically
    )

    # 5. Final evaluation
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_acc = evaluate_sft(trained_model, test_loader, device=str(device))
    print(f"Final Test Accuracy: {test_acc:.4f}")
