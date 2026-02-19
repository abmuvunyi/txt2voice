import argparse
import time
import torch
from transformers import VitsModel, VitsTokenizer
import scipy.io.wavfile
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/mms-tts-eng")
    parser.add_argument("--text", type=str, default="Hello, how can I help you today?")
    parser.add_argument("--output_file", type=str, default="output.wav")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if torch.backends.mps.is_available() and not args.cpu:
        device = "mps"
    
    print(f"Loading model {args.model_name} on {device}...")
    model = VitsModel.from_pretrained(args.model_name).to(device)
    tokenizer = VitsTokenizer.from_pretrained(args.model_name)

    text = args.text
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"Synthesizing: '{text}'")
    
    # Warmup
    with torch.no_grad():
        _ = model(**inputs).waveform

    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        output = model(**inputs).waveform
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"Latency: {latency_ms:.2f} ms")
    
    # Save audio
    waveform = output.cpu().numpy().squeeze()
    # Normalize if needed, but usually it's fine. MMS output is roughly -1 to 1.
    
    # Sampling rate for MMS is usually 16kHz or defined in config
    sampling_rate = model.config.sampling_rate
    print(f"Sampling rate: {sampling_rate}")
    
    scipy.io.wavfile.write(args.output_file, rate=sampling_rate, data=waveform)
    print(f"Audio saved to {args.output_file}")
    
    # Size check
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size in memory: {size_all_mb:.2f} MB")

if __name__ == "__main__":
    main()
