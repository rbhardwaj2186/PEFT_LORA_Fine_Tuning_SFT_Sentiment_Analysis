# Efficient Sentiment Analysis with PEFT and LoRA




https://github.com/user-attachments/assets/b5a10bc3-0f6d-4bf5-afad-7dc42d1ef226



https://github.com/user-attachments/assets/b33e8984-a860-48a3-88b2-e075c726ed9e



A parameter-efficient approach to sentiment analysis using DistilBERT and Low-Rank Adaptation. This project demonstrates how to achieve high-accuracy sentiment classification while significantly reducing computational costs and model size.

## Features

- Parameter-Efficient Fine-Tuning with LoRA adaptation
- Preferential training with high-quality example pairs
- Quantization for reduced memory footprint
- Optimized for deployment across various platforms
- Built on DistilBERT for balanced efficiency and accuracy

## Results

- 85% classification accuracy on IMDB dataset
- 40% reduction in model size
- 30% lower computational costs
- 60% faster training cycles

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```python
from orchestrator.pipeline_sft import run_sft_pipeline

run_sft_pipeline(use_peft=True)
```

## Project Structure

```
sentiment_analysis/
├── data_service/
│   ├── data_loader.py       # Dataset handling
│   └── data_preprocessor.py # Text preprocessing
├── model_service/
│   ├── model_definition.py  # DistilBERT + LoRA setup
│   └── train_sft.py        # Training pipeline
└── orchestrator/
    ├── config.py           # Configuration
    └── pipeline_sft.py     # Main execution flow
```

## Usage

### Training

```python
# Initialize model with PEFT
model = create_sft_model(
    model_name="distilbert-base-uncased",
    use_peft=True
)

# Train with preference data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_train_ds,
    eval_dataset=eval_dataset
)

trainer.train()
```

### Inference

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./results")
sentiment = model.predict("Great movie, highly recommended!")
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- PEFT 0.2+
- Accelerate 0.17+

## License

MIT

## Contact

Email: your.email@domain.com
GitHub: [Your Profile]
