# main.py

import argparse
from orchestrator.pipeline_sft import run_sft_pipeline

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDB with PEFT options")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to training state checkpoint file (e.g., ./checkpoints/training_state_epoch_3.pt)")
    parser.add_argument("--use_peft", dest="use_peft", action="store_true", default=True, help="Enable PEFT for training")
    parser.add_argument("--no-use_peft", dest="use_peft", action="store_false", help="Disable PEFT for training")
    args = parser.parse_args()

    if args.resume and not args.resume_checkpoint:
        print("[Error] --resume flag is set but no --resume_checkpoint provided.")
        exit(1)

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume_checkpoint}")
    else:
        print("Starting training from scratch.")
    
    print(f"PEFT enabled: {args.use_peft}")

    run_sft_pipeline(
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint if args.resume else None,
        use_peft=args.use_peft
    )

if __name__ == "__main__":
    main()
