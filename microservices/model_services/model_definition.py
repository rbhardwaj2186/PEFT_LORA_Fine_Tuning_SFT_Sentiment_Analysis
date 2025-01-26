# microservices/model_services/model_definition.py

from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

def create_sft_model(model_name="distilbert-base-uncased", num_labels=2, use_peft=False):
    """
    Loads model with optional PEFT (LoRA) for sequence classification.
    
    Args:
        model_name (str): Pretrained model name.
        num_labels (int): Number of classification labels.
        use_peft (bool): Whether to apply PEFT (LoRA) for fine-tuning.
    
    Returns:
        model: The configured model ready for training or inference.
    """
    # Load the base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    if use_peft:
        # LoRA Configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_lin", "v_lin"],  # Must match actual module names in DistilBERT
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS"
        )
        
        # Apply LoRA to the base model
        peft_model = get_peft_model(base_model, lora_config)
        
        return peft_model
    
    return base_model
