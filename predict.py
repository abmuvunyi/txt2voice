import os
import sys
import torch
import time
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import VitsModel, VitsTokenizer

def run_inference(text, model_path="facebook/mms-tts-kin", output_path="output.wav"):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model-tokenizer from local or HF
    if os.path.exists("./models/best_model_finetuned"):
        model_path = "./models/best_model_finetuned"
    
    print(f"Loading model: {model_path}...")
    model = VitsModel.from_pretrained(model_path).to(device)
    tokenizer = VitsTokenizer.from_pretrained(model_path)
    
    print(f"Synthesizing: \"{text}\"")
    start_time = time.time()
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
    
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
