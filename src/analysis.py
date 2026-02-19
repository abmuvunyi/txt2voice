import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from datasets import load_dataset
from collections import Counter

def analyze_dataset(dataset):
    """
    Analyzes audio dataset for duration, counts, and stats.
    """
    print("--- Starting Data Analysis ---")
    
    durations = []
    speaker_ids = []
    
    for i, item in enumerate(dataset):
        audio = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        
        # Calculate duration
        dur_sec = len(audio) / sampling_rate
        durations.append(dur_sec)
        
        # Track speakers
        if 'speaker_id' in item:
            speaker_ids.append(item['speaker_id'])
            
        if i % 1000 == 0:
            print(f"Analyzed {i} samples...", end='\r')
            
    durations = np.array(durations)
    
    stats = {
        "Total Clips": len(durations),
        "Total Hours": np.sum(durations) / 3600,
        "Mean Duration (s)": np.mean(durations),
        "Min Duration (s)": np.min(durations),
        "Max Duration (s)": np.max(durations),
        "Unique Speakers": len(set(speaker_ids)) if speaker_ids else 0
    }
    
    print("\n--- Analysis Results ---")
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")
        
    return durations, speaker_ids, stats

def plot_distributions(durations, speaker_ids=None):
    """
    Plots duration histogram and speaker distribution.
    """
    fig, axes = plt.subplots(1, 2 if speaker_ids else 1, figsize=(15, 6))
    
    # 1. Duration Histogram
    ax_dur = axes[0] if speaker_ids else axes
    ax_dur.hist(durations, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax_dur.set_title("Audio Clip Duration Distribution", fontweight='bold')
    ax_dur.set_xlabel("Duration (seconds)")
    ax_dur.set_ylabel("Frequency")
    ax_dur.grid(True, linestyle='--', alpha=0.5)
    
    # 2. Speaker Bar Chart
    if speaker_ids:
        counts = Counter(speaker_ids)
        sorted_keys = sorted(counts.keys())
        ax_spk = axes[1]
        ax_spk.bar([str(k) for k in sorted_keys], [counts[k] for k in sorted_keys], color='#e74c3c', alpha=0.7)
        ax_spk.set_title("Speaker Contribution Layout", fontweight='bold')
        ax_spk.set_xlabel("Speaker ID")
        ax_spk.set_ylabel("Clip Count")
        ax_spk.grid(True, axis='y', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.show()

def plot_spectrogram(audio_array, sr=16000, title="Audio Spectrogram"):
    """
    Plots a professional log-mel spectrogram.
    """
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    plt.show()
