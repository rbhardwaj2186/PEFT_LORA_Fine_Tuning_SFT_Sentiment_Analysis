import argparse
from orchestrator.pipeline_sft import run_sft_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on IMDB with optional checkpointing.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, help="Path to .pt checkpoint file")

    args = parser.parse_args()
    # Add this for testing
    print(f"Resume Training: {args.resume}")
    print(f"Checkpoint Path: {args.resume_checkpoint}")
    
    run_sft_pipeline(
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint
    )

      