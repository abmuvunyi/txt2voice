import streamlit as st
import torch
import numpy as np
import time
import os
import io
import scipy.io.wavfile as wavfile
from transformers import VitsModel, VitsTokenizer

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Tekana TTS",
    page_icon="🗣️",
    layout="centered"
)

# ── Model Loading (cached) ─────────────────────────────────
@st.cache_resource
def load_model():
    finetuned = "./models/best_model_finetuned"
    base = "facebook/mms-tts-kin"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = finetuned if os.path.exists(finetuned) else base

    try:
        model = VitsModel.from_pretrained(model_id).to(device)
        tokenizer = VitsTokenizer.from_pretrained(model_id)
    except Exception:
        model = VitsModel.from_pretrained(base).to(device)
        tokenizer = VitsTokenizer.from_pretrained(base)

    model.eval()
    return model, tokenizer, device, model_id

model, tokenizer, device, model_id = load_model()

# ── UI ─────────────────────────────────────────────────────
st.title("🗣️ Tekana — Kinyarwanda TTS")
st.markdown("Type Kinyarwanda text and hear it spoken aloud. Built for the Tekana Health Line.")

st.divider()

# Input
text = st.text_area(
    "Enter Kinyarwanda text",
    placeholder="Muraho, nagufasha gute uyu munsi?",
    height=100
)

col1, col2 = st.columns([1, 3])
with col1:
    generate = st.button("🔊 Generate", type="primary", use_container_width=True)

# Examples
st.markdown("**Quick examples:**")
examples = [
    "Muraho, nagufasha gute uyu munsi?",
    "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha.",
    "Ni ngombwa ko ubonana umuganga vuba.",
    "Twabanye nawe kandi tuzakomeza kukwitaho.",
    "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."
]
for ex in examples:
    if st.button(ex, key=ex):
        text = ex
        generate = True

st.divider()

# Synthesis
if generate and text.strip():
    with st.spinner("Synthesizing..."):
        t0 = time.time()
        inputs = tokenizer(text.strip(), return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform
        latency = (time.time() - t0) * 1000

    audio_np = output.cpu().numpy().squeeze()

    # Write to in-memory WAV
    buf = io.BytesIO()
    audio_int16 = (audio_np / np.max(np.abs(audio_np)) * 32767).astype(np.int16)
    wavfile.write(buf, 16000, audio_int16)

    st.audio(buf, format="audio/wav")
    st.success(f"✅ Latency: {latency:.0f} ms")

# Footer
st.divider()
st.caption(f"Model: `{model_id}` · Device: `{device}`")
