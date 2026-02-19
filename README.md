# Tekana Health — Kinyarwanda TTS

Production-ready Text-to-Speech pipeline for the Tekana medical health line,
built on Meta's MMS-VITS architecture with native Kinyarwanda support.

| Metric | Value |
|:---|:---|
| Latency | ~300 ms |
| Model Size | 137 MB |
| Architecture | VITS (end-to-end) |
| Language | Kinyarwanda |

## Quick Start

### Local
```bash
pip install -r requirements.txt
python predict.py "Muraho, nagufasha gute uyu munsi?"
```

### Docker
```bash
docker build -t tekana-tts .
docker run tekana-tts "Muraho, nagufasha gute uyu munsi?"
```

### Interactive Demo (Gradio)
```bash
pip install -r requirements.txt
python app.py
# Opens at http://localhost:7860
```

### Streamlit Demo
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

## Project Structure

```
├── Solution_Notebook.ipynb   # Main notebook (EDA → Training → Inference)
├── WRITEUP.md                # Technical write-up
├── predict.py                # CLI inference script
├── app.py                    # Gradio interactive demo
├── streamlit_app.py          # Streamlit interactive demo
├── Dockerfile                # Container deployment
├── requirements.txt          # Python dependencies
├── src/
│   ├── train.py              # Training logic (used by notebook)
│   ├── preprocess.py         # Text normalization & filtering
│   ├── analysis.py           # Dataset analysis utilities
│   ├── inference.py          # Inference utilities
│   ├── dataset.py            # Dataset loading utilities
│   └── utils.py              # Seed & timing helpers
├── output_final/             # Generated mandatory assessment audio
│   ├── mandatory_0.wav
│   ├── mandatory_1.wav
│   ├── mandatory_2.wav
│   ├── mandatory_3.wav
│   └── mandatory_4.wav
└── data/                     # Dataset cache (not in repo)
```

## Approach

**Single-Speaker Precision Fine-Tuning**: Starting from `facebook/mms-tts-kin`,
freeze the VITS decoder to preserve audio quality, then fine-tune acoustic
encoders on Speaker 3's data for persona adaptation.

See [WRITEUP.md](WRITEUP.md) for the full technical explanation.

## Mandatory Sentences

The following sentences are synthesized in the notebook and saved to `output_final/`:

1. "Muraho, nagufasha gute uyu munsi?"
2. "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha."
3. "Ni ngombwa ko ubonana umuganga vuba."
4. "Twabanye nawe kandi tuzakomeza kukwitaho."
5. "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."
