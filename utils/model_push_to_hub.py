import argparse
import logging
import importlib
from transformers import AutoConfig, AutoTokenizer, AutoProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("model_push_to_hub")


def push_to_hub():
    parser = argparse.ArgumentParser(description="Push locally saved training output to Hugging Face Hub.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the local checkpoint directory.")
    parser.add_argument("--hub_model_id", type=str, required=True, help="Hugging Face repo ID (e.g., 'username/model_name').")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face authentication token.")
    parser.add_argument("--private", action="store_true", help="Whether the repository should be private.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True, help="Whether to trust remote code.")

    args = parser.parse_args()

    logger.info(f"Loading config from {args.ckpt_path}...")
    config = AutoConfig.from_pretrained(args.ckpt_path, trust_remote_code=args.trust_remote_code)

    # Determine the model class based on config
    arch = config.architectures[0] if config.architectures else "Unknown"

    # import arch and instantiate model accordingly
    module = importlib.import_module("transformers")
    model_class = getattr(module, arch, None)

    if model_class is None:
        logger.error(f"Model architecture {arch} not found in transformers library.")
        raise ValueError(f"Model architecture {arch} not found in transformers library.")

    logger.info(f"Detected model class: {model_class.__name__}")

    logger.info(f"Loading model from {args.ckpt_path}...")
    model = model_class.from_pretrained(args.ckpt_path, config=config, trust_remote_code=args.trust_remote_code, dtype="auto", device_map="cpu")

    # Push Processor
    try:
        logger.info("Attempting to load processor...")
        processor = AutoProcessor.from_pretrained(args.ckpt_path, trust_remote_code=args.trust_remote_code)
        logger.info("Processor loaded successfully.")
        logger.info(f"Pushing processor to hub: {args.hub_model_id}...")
        processor.push_to_hub(args.hub_model_id, token=args.token, private=args.private)
        logger.info("Processor pushed to hub successfully.")
    except Exception as e:
        logger.warning(f"Processor not found or failed to push: {e}. Skipping processor.")

    # Push Tokenizer
    try:
        logger.info("Attempting to load tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path, trust_remote_code=args.trust_remote_code)
        logger.info("Tokenizer loaded successfully.")
        logger.info(f"Pushing tokenizer to hub: {args.hub_model_id}...")
        tokenizer.push_to_hub(args.hub_model_id, token=args.token, private=args.private)
        logger.info("Tokenizer pushed to hub successfully.")
    except Exception as e:
        logger.warning(f"Tokenizer not found or failed to push: {e}. Skipping tokenizer.")

    # Push Model
    logger.info(f"Pushing model weights to hub: {args.hub_model_id}...")
    model.push_to_hub(args.hub_model_id, token=args.token, private=args.private, safe_serialization=True)
    logger.info("Model weights pushed to hub successfully.")
    logger.info("All components processed.")


if __name__ == "__main__":
    push_to_hub()
