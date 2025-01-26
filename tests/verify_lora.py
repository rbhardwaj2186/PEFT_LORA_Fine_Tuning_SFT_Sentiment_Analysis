# Fine_tuning_projectg/tests/verify_lora.py

from microservices.model_services.model_definition import create_sft_model

def verify_lora_injection():
    model = create_sft_model(use_peft=True)
    # Check for LoRA modules
    lora_found = False
    for name, module in model.named_modules():
        if "lora" in name.lower():
            print(f"LoRA adapter found in module: {name}")
            lora_found = True
    if not lora_found:
        print("No LoRA adapters found. Please check your configuration.")

if __name__ == "__main__":
    verify_lora_injection()
