import sys
import torch
import time
import numpy as np
import scipy.io.wavfile as wavfile
from src.model_utils import load_tts_model


def run_inference(text, model_path=None, output_path="output.wav"):
    # Load model-tokenizer from local or HF
    model, tokenizer, device, model_id = load_tts_model(model_path)

    print(f"Synthesizing: \"{text}\"")
    start_time = time.time()

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device))

    latency = (time.time() - start_time) * 1000

    audio = output.waveform.cpu().numpy().squeeze()
    # Normalize and convert to int16
    audio = (audio / np.max(np.abs(audio)) * 32767).astype(np.int16)

    wavfile.write(output_path, 16000, audio)

    print(f"Success! Saved to {output_path}")
    print(f"Latency: {latency:.2f} ms")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_text = "Muraho, nagufasha gute uyu munsi?"
    else:
        test_text = sys.argv[1]

    run_inference(test_text)
