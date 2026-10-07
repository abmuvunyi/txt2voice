import gradio as gr
import torch
import time
import os
from src.model_utils import load_tts_model, get_device

# ── Model Loading ──────────────────────────────────────────
device = get_device()
model, tokenizer, device, model_id = load_tts_model(device=device)
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
    gr.Markdown(
        "Type Kinyarwanda text below and click **Generate** to hear synthesized speech.")

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

    btn.click(
        synthesize,
        inputs=text_input,
        outputs=[
            audio_output,
            latency_label])

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
    host = os.getenv("HOST", "127.0.0.1")
    demo.launch(server_name=host, server_port=7860)
