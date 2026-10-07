import os
import torch
from transformers import VitsModel, VitsTokenizer


def get_device(force_cpu=False):
    """Returns the optimal PyTorch device."""
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tts_model(
        model_path=None,
        base_model="facebook/mms-tts-kin",
        device=None):
    """
    Loads a VITS model and tokenizer from the given path, falling back to a base model.
    """
    if device is None:
        device = get_device()

    finetuned_path = "./models/best_model_finetuned"

    if model_path is None:
        if os.path.exists(finetuned_path):
            model_path = finetuned_path
        else:
            model_path = base_model

    print(f"Loading model: {model_path} on {device}...")
    try:
        model = VitsModel.from_pretrained(model_path).to(device)  # nosec B615
        tokenizer = VitsTokenizer.from_pretrained(model_path)  # nosec B615
    except Exception as e:
        print(
            f"Failed to load {model_path} ({e}), falling back to {base_model}...")
        model = VitsModel.from_pretrained(base_model).to(device)  # nosec B615
        tokenizer = VitsTokenizer.from_pretrained(base_model)  # nosec B615
        model_path = base_model

    model.eval()
    return model, tokenizer, device, model_path
