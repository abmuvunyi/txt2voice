# Tekana TTS — Technical Write-Up

**Candidate**: [Your Name]  
**Date**: February 2025  
**Challenge**: Kinyarwanda Text-to-Speech for the Tekana Health Line

---

## 1. Background & My Approach

Tekana is a voice-based system that helps young Rwandans with health questions. It already handles speech-to-text and response generation — the missing piece is turning that response back into spoken Kinyarwanda. That's what this project tackles.

Going into this, I knew two things would matter more than anything: the voice has to sound **trustworthy** (not robotic, not foreign), and the system has to be **fast enough** for a real phone call (<800 ms). Everything else — architecture, training strategy, data choices — follows from those two constraints.

---

## 2. Choosing the Base Model

### What I Picked

I went with **`facebook/mms-tts-kin`** — Meta's MMS model with a VITS checkpoint specifically for Kinyarwanda.

### What Else I Looked Into

Before committing, I spent time evaluating alternatives:

- **Coqui TTS (VITS from scratch)**: Flexible and well-documented, but there's no Kinyarwanda checkpoint. I'd be training pronunciation from zero, which realistically needs 40+ hours of clean single-speaker data and days of GPU time. With ~14 hours of multi-speaker data and a 3-5 day window, that felt like a gamble I couldn't afford.

- **Piper TTS**: Extremely fast and lightweight, which is appealing for the latency constraint. But again — no Kinyarwanda support. I'd need to first train an ONNX model from scratch, which has the same data/time problem as Coqui.

- **Training VITS from scratch on the provided data**: I seriously considered this because it would give me full control over the voice. But VITS training typically needs 200k+ steps to converge. On my hardware (Mac M-series, no dedicated GPU), that's simply not feasible in the given timeframe.

### Why MMS-VITS Won

It came down to a simple question: *which approach gives the best voice quality within the constraints?*

MMS-VITS already "speaks" Kinyarwanda — it was pre-trained on 1,100+ languages including Kinyarwanda. That means pronunciation, prosody, and basic naturalness are already solved. I just need to adapt the *persona* (voice identity) to match the Tekana health line character. That's a much smaller problem than teaching a model an entire language.

The numbers work too: 137 MB model size (under the 200 MB limit) and ~300 ms inference latency (well under 800 ms).

---

## 3. How I Prepared the Data

### What I Did

The dataset comes pre-processed, which is nice, but I still needed to validate and filter it:

1. **Duration filtering (2–12s)**: I removed clips shorter than 2 seconds (not enough phonetic content for the model to learn from) and longer than 12 seconds (memory issues during training, and longer clips tend to have inconsistent prosody). This is specified in the brief.

2. **Sample rate check**: Verified everything was at 16 kHz. It was — no resampling needed. This matters because the VITS model expects exactly 16 kHz input.

3. **Text normalization**: Lowercased text and removed punctuation to match what the MMS tokenizer expects. The tokenizer uses character-level encoding, so inconsistent casing would confuse it.

4. **Speaker filtering**: For fine-tuning, I isolated Speaker 3's clips specifically (more on why below).

### What I Found in the Data

The EDA revealed a few things worth noting:

- **Uneven speaker split**: The three speakers don't contribute equally. Some have significantly more clips than others. If I trained on all of them, the model would learn a weird average voice that doesn't sound like anyone in particular. This pushed me toward single-speaker training.

- **Variable recording quality**: Looking at the spectrograms, some clips have noticeably higher noise floors than others. A few show what looks like clipping. I didn't throw these out (the dataset is already limited), but the frozen-decoder strategy I describe below was partly motivated by this — it prevents noisy training data from degrading the model's audio generation quality.

- **Some very short clips**: A handful of clips under 1 second. These got caught by the 2-second floor filter.

I picked **Speaker 3** because after listening to samples from each speaker and examining spectrograms, Speaker 3's recordings had the most consistent quality and the clearest articulation. For a health line that needs to inspire trust, voice clarity isn't optional.

---

## 4. Fine-Tuning vs. Training from Scratch

I chose **fine-tuning**, and honestly it wasn't a close call. Here's my reasoning:

**Data**: 14 hours across 3 speakers means roughly 4-5 hours per speaker. VITS trained from scratch typically needs 20-50 hours of single-speaker data to sound natural. I'm well short of that.

**Time**: Even if I had enough data, training VITS to convergence takes 200,000+ steps. On a cloud GPU that's 2-3 days. On my local machine, it's not happening.

**Risk management**: With fine-tuning, the worst case is the model sounds exactly like the base MMS model — still perfectly intelligible Kinyarwanda. With training from scratch, the worst case is unintelligible noise. For a health application, I'd rather have a solid floor.

The one downside of fine-tuning is less control over the final voice. But since the brief says "a perfect accent is not required, but it should not feel foreign or robotic," the MMS base voice already clears that bar.

---

## 5. Training Decisions

### The "Single-Speaker Precision" Strategy

This was the most important design decision in the whole project, and it took a few failed attempts to land on it.

**The idea**: Freeze the VITS decoder entirely. Only train the acoustic encoders (flow network, posterior encoder, text encoder). This adapts the *voice identity* without touching the component that actually generates the audio waveform.

**Why it works**: The VITS decoder (HiFi-GAN) was pre-trained on thousands of hours of clean audio. It knows how to produce clean waveforms. When I tried unfreezing it in my first attempt (v1), it started "forgetting" this knowledge after just a few hundred steps — the result was metallic, chattery audio. Freezing it guarantees the audio quality never drops below the pre-trained baseline.

### Hyperparameters

| Parameter | Value | Why |
|:---|:---|:---|
| Learning Rate | 1e-4 | This is higher than typical fine-tuning (usually 1e-5 to 5e-5). I can afford a higher rate because only ~25% of parameters are being updated — the risk of catastrophic forgetting is lower with a frozen decoder. |
| Batch Size | 2 | Hardware constraint (Apple MPS memory). Compensated with gradient accumulation. |
| Gradient Accumulation | 4 steps | Effective batch size of 8, which gives stable gradient estimates without blowing up memory. |
| Epochs | 1 | Just one pass through the data. With TTS models, overfitting manifests as "mode collapse" — the model memorizes exact utterances instead of learning general speech patterns. One epoch is conservative but safe. |
| Gradient Clipping | 1.0 | The MPS backend can produce gradient spikes. Clipping at 1.0 prevents these from destabilizing training. |

### Loss Function

I used **Mel-Spectrogram L1 Loss** instead of the standard VITS waveform-level losses. This was a deliberate choice born from a painful lesson:

In v1, I used the standard waveform L1 loss. The spectrograms looked good, but the audio had an awful metallic chattering sound. The problem: waveform L1 penalizes *phase* differences between predicted and target audio. Two waveforms can sound identical to a human but have completely different phases — the loss treats them as errors and tries to "fix" them, introducing artifacts.

Mel-spectrogram loss operates in the frequency domain, where phase doesn't matter. It asks: "do these two sounds have the same energy at the same frequencies?" That's much closer to what humans actually hear.

### When I Stopped Training

I monitored two things during training:
- **Log-Spectral Distance (LSD)**: Should decrease. Measures how close the generated speech is to the target in spectral space.
- **Cosine Similarity**: Should increase. Measures overall spectral shape agreement.

I also synthesized the five mandatory test sentences at checkpoints and listened to them. If they sounded worse than the previous checkpoint, that's a signal to stop regardless of what the numbers say. For TTS, your ears are the ultimate metric.

---

## 6. Evaluation

### Why My Evaluation Approach Fits This Use Case

Tekana serves young people seeking health guidance — often on sensitive topics. If the voice sounds *off* in any way (robotic, garbled, mispronouncing), callers will lose trust and hang up. Because of this, I weighted subjective evaluation heavily alongside quantitative metrics.

### Metrics

| Metric | What It Tells Me |
|:---|:---|
| **Inference Latency** | Is this fast enough for a real-time phone call? Target: <800 ms. Achieved: ~300 ms. |
| **Log-Spectral Distance (LSD)** | How spectrally accurate is the generated speech? Lower = closer to natural speech. |
| **Cosine Similarity** | Does the overall spectral shape match the target speaker? |
| **Subjective Listening** | Does it *sound* right? Clear, natural, consistent? This is the most important one for a health application. |

### Mandatory Sentences

I synthesized all five required sentences:

1. "Muraho, nagufasha gute uyu munsi?"
2. "Niba ufite ibibazo bijyanye n'ubuzima bwawe, twagufasha."
3. "Ni ngombwa ko ubonana umuganga vuba."
4. "Twabanye nawe kandi tuzakomeza kukwitaho."
5. "Ushobora kuduhamagara igihe cyose ukeneye ubufasha."

All generated with latency under 400 ms. Audio files are in `./output_final/` and also play inline in the notebook.

My subjective assessment: the base MMS model produces clear, intelligible Kinyarwanda. It doesn't perfectly match any of the three speakers in the dataset (that would require significantly more fine-tuning data), but it sounds natural and not robotic — which is the actual requirement.

---

## 7. Production Packaging

### What's Included

| File | What It Does |
|:---|:---|
| `predict.py` | Takes a text string as input, outputs a WAV file, prints latency. Run: `python predict.py "Your text here"` |
| `Dockerfile` | One-command deployment: `docker build -t tekana-tts . && docker run tekana-tts "Muraho!"` |
| `requirements.txt` | Pinned dependencies for reproducibility. |
| `app.py` | Gradio-based interactive demo with a text box and audio playback. Run: `python app.py` |

### Inference Logic

The inference script is smart about model selection: it checks if a fine-tuned model exists locally (`./models/best_model_finetuned`). If found, it runs a quality gate — basic sanity checks on the output — before using it. If the fine-tuned model doesn't exist or fails the gate, it falls back to the base `facebook/mms-tts-kin` model. This means the system always produces usable output.

---

## 8. What Went Wrong (Experiment History)

I think it's worth being honest about what didn't work, because the failures shaped the final solution.

| Attempt | What I Tried | What Happened |
|:---|:---|:---|
| **v1** | Unfroze everything (36M params), trained on all speakers, waveform L1 loss | Horrible metallic noise. Phase mismatch between predicted and target waveforms created audible artifacts. |
| **v2** | Switched to mel-only loss, unfroze decoder with lower LR | Better spectrally, but the decoder started drifting — audio quality degraded over training. Also hit disk space issues from MPS cache buildup. |
| **v3** | Froze decoder completely, Speaker 3 only, mel L1 loss | Clean output. The frozen decoder preserved pre-trained audio quality while the encoders adapted to the target persona. This is the final approach. |

The key lesson: **don't touch the decoder unless you have enough data to retrain it properly.** The VITS decoder is a HiFi-GAN that was trained on thousands of hours of clean audio. Updating it on 4 hours of potentially noisy data causes it to "forget" how to generate clean waveforms.

---

## 9. Use of Generative AI Tools

I used an **AI coding assistant** (Antigravity) during this project. Here's a transparent breakdown of what it helped with:

### What AI Assisted With (~10-15% of work)

- **Dockerfile boilerplate**: Generated the initial Dockerfile structure and system dependency list. I reviewed and adjusted the final version.
- **Notebook formatting**: Helped structure markdown cells and format tables consistently across the notebook sections.
- **Requirements.txt**: Generated the initial dependency list based on imports used in the project.
- **Utility code structure**: Suggested the `seed_everything()` pattern and the `@dataclass` config pattern. Standard boilerplate I've used before, but faster to generate than type out.
- **Write-up drafting**: Used for an initial draft of this write-up, which I then rewrote and edited to reflect my actual experience and voice.

### What I Did Myself

- **All architectural decisions**: Model selection, freezing strategy, loss function choice, speaker selection — these came from reading the VITS paper, understanding the MMS architecture, and iterating through failed experiments.
- **Debugging**: The phase-chatter issue (v1), the disk space problem (MPS cache), the decoder drift (v2) — these required understanding what was actually happening at a signal-processing level.
- **EDA and data analysis**: Interpreting spectrograms, choosing Speaker 3 based on quality, identifying the uneven speaker distribution.
- **Evaluation and subjective listening**: No AI can tell me if a Kinyarwanda sentence sounds natural. That required listening, comparing, and making judgment calls.
- **Training loop design**: The specific combination of frozen decoder + mel-only loss + single speaker was my solution to problems I encountered hands-on.

---

## 10. References

1. **VITS** — Kim, J., Kong, J., & Son, J. (2021). *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.* Proceedings of ICML 2021. [arXiv:2106.06103](https://arxiv.org/abs/2106.06103)
   — The core architecture behind our TTS model. VITS combines a variational autoencoder, normalizing flows, and an adversarial training objective to produce high-quality speech in a single end-to-end model (no separate acoustic model + vocoder pipeline). This is why inference is fast: one forward pass from text to waveform.

2. **MMS (Massively Multilingual Speech)** — Pratap, V., et al. (2023). *Scaling Speech Technology to 1,000+ Languages.* Meta AI Research. [arXiv:2305.13516](https://arxiv.org/abs/2305.13516)
   — The multilingual framework that provides the Kinyarwanda checkpoint (`facebook/mms-tts-kin`). MMS uses wav2vec 2.0-style pre-training across 1,100+ languages, then adapts VITS for TTS in each language. This is what gives us a working Kinyarwanda voice out of the box — the model already learned Kinyarwanda phonology from the MMS pre-training data.

3. **HiFi-GAN** — Kong, J., Kim, J., & Bae, J. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* Proceedings of NeurIPS 2020. [arXiv:2010.05646](https://arxiv.org/abs/2010.05646)
   — The vocoder (decoder) inside VITS. HiFi-GAN is the component I chose to freeze during fine-tuning. It converts mel spectrograms into raw audio waveforms using transposed convolutions and multi-scale discriminators. Understanding this architecture is why I knew freezing it was safe — it was trained to be a universal waveform generator, not a speaker-specific one.

4. **Mel-Spectrogram Loss for TTS** — Shen, J., et al. (2018). *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.* Proceedings of ICASSP 2018. [arXiv:1712.05884](https://arxiv.org/abs/1712.05884)
   — While this paper is about Tacotron 2 (not VITS), it established the mel-spectrogram L1 loss as a standard TTS training objective. I adopted this loss function instead of waveform-level loss because it's phase-invariant — it measures perceptual similarity without penalizing harmless phase differences that cause audible artifacts.
