import gradio as gr
from transformers import VitsModel, VitsTokenizer
import torch
import numpy as np
import time
import os

# ── Model Loading ──────────────────────────────────────────
FINETUNED_PATH = "./models/best_model_finetuned"
BASE_MODEL     = "facebook/mms-tts-kin"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Model...")
if os.path.exists(FINETUNED_PATH):
    print(f"  Found fine-tuned model at {FINETUNED_PATH}")
    model_id = FINETUNED_PATH
else:
    print(f"  Using base model: {BASE_MODEL}")
    model_id = BASE_MODEL

try:
    model = VitsModel.from_pretrained(model_id).to(device)
    tokenizer = VitsTokenizer.from_pretrained(model_id)
except Exception as e:
    print(f"  Fallback triggered ({e})")
    model = VitsModel.from_pretrained(BASE_MODEL).to(device)
    tokenizer = VitsTokenizer.from_pretrained(BASE_MODEL)

model.eval()
print("  ✅ Model ready!\n")

# ── Inference Function ─────────────────────────────────────
def synthesize(text):
    if not text.strip():
        return None, "Please enter some Kinyarwanda text."

    t0 = time.time()
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs).waveform

    latency = (time.time() - t0) * 1000
    audio_np = output.cpu().numpy().squeeze()

    return (16000, audio_np), f"✅ Latency: {latency:.0f} ms"

# ── Gradio UI ──────────────────────────────────────────────
with gr.Blocks(
    title="Tekana TTS Demo",
    css="#component-0 {max-width: 800px; margin: auto;}"
) as demo:
    gr.Markdown("# 🗣️ Tekana — Kinyarwanda TTS")
    gr.Markdown("Type Kinyarwanda text below and click **Generate** to hear synthesized speech.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text (Kinyarwanda)",
                placeholder="Muraho, nagufasha gute uyu munsi?",
                lines=3
            )
            btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Synthesized Speech")
            latency_label = gr.Label(label="Performance")

    btn.click(synthesize, inputs=text_input, outputs=[audio_output, latency_label])

    gr.Examples(
        examples=[
            "Muraho, nagufasha gute uyu munsi?",
            "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
            "Ni ngombwa ko ubonana umuganga vuba.",
            "Twabanye nawe kandi tuzakomeza kukwitaho.",
            "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."
        ],
        inputs=text_input
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
