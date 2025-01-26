# microservices/model_services/train_sft.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from peft import PeftModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup  # Optional: For learning rate scheduling

def save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define directories for saving PEFT model and training state
    peft_model_dir = os.path.join(checkpoint_dir, f"peft_model_epoch_{epoch}")
    training_state_path = os.path.join(checkpoint_dir, f"training_state_epoch_{epoch}.pt")
    
    # Save the PEFT model using save_pretrained
    model.save_pretrained(peft_model_dir)
    print(f"[Checkpoint] Saved PEFT model at: {peft_model_dir}")
    
    # Save training state (optimizer state, epoch, average loss)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_loss': avg_train_loss,
        'peft_model_dir': peft_model_dir
    }, training_state_path)
    print(f"[Checkpoint] Saved training state at: {training_state_path}")

def load_checkpoint(checkpoint_file, device="cuda"):
    print(f"[Checkpoint] Loading training state from {checkpoint_file}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    epoch = checkpoint['epoch']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    avg_train_loss = checkpoint['avg_train_loss']
    peft_model_dir = checkpoint['peft_model_dir']
    
    print(f"[Checkpoint] Resumed from epoch {epoch}")
    print(f"[Checkpoint] PEFT model directory: {peft_model_dir}")
    
    return peft_model_dir, optimizer_state_dict, epoch, avg_train_loss

def evaluate_sft(model, dataloader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_sft(model, 
              train_ds, 
              test_ds,
              device="cuda", 
              epochs=3, 
              batch_size=16, 
              lr=2e-5,
              checkpoint_dir="./checkpoints",
              resume=False,
              resume_checkpoint=None,
              use_peft=False):
    """
    Fine-tunes the classification model (SFT) with optional checkpointing.
    
    Args:
      model: The model to train.
      train_ds: Training dataset.
      test_ds: Test dataset.
      device (str): Device to train on ('cuda' or 'cpu').
      epochs (int): Number of training epochs.
      batch_size (int): Batch size for training.
      lr (float): Learning rate.
      checkpoint_dir (str): Directory to save checkpoints. If None, no checkpoints are saved.
      resume (bool): Whether to resume from a given checkpoint file.
      resume_checkpoint (str): Path to a training state checkpoint file to resume from.
      use_peft (bool): Whether to use PEFT during training.
    """
    # 1. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Move model to device
    model.to(device)
    print(f"Model moved to device: {device}")
    
    # 3. Prepare optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    # Optional: Prepare a learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=int(0.1 * total_steps), 
                                                num_training_steps=total_steps)
    
    # 4. If resuming, load checkpoint
    start_epoch = 0
    if resume and resume_checkpoint is not None and os.path.exists(resume_checkpoint):
        peft_model_dir, optimizer_state_dict, start_epoch, avg_train_loss = load_checkpoint(resume_checkpoint, device=device)
        
        # Load the PEFT model from the saved directory
        model = PeftModelForSequenceClassification.from_pretrained(peft_model_dir, device_map={"": device}).to(device)
        print("Loaded PEFT model from checkpoint.")
        
        # Load the optimizer state
        optimizer.load_state_dict(optimizer_state_dict)
        print("Loaded optimizer state from checkpoint.")
    
    # 5. Diagnostic: Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found. Ensure that PEFT is correctly integrated and parameters are not all frozen.")
    
    # 6. Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Check if loss requires grad
            if not loss.requires_grad:
                raise RuntimeError("Loss does not require gradients. Check if model parameters require gradients.")

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()  # Optional: Update learning rate

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc = evaluate_sft(model, test_loader, device=device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save checkpoint after each epoch
        if checkpoint_dir is not None:
            save_checkpoint(model, optimizer, epoch+1, avg_train_loss, checkpoint_dir)

    return model
