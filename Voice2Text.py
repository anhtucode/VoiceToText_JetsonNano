import os
import sys
import time
import psutil
import torch
import torchaudio
import librosa
import numpy as np
from df.enhance import (
    enhance,
    init_df,
    load_audio,
    save_audio
)
from pathlib import Path
from utils_vad import get_speech_timestamps
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import pipeline
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering

# =========================
# Công cụ đo lường Performance (Latency, RAM, VRAM)
# =========================
class PerfMonitor:
    def __init__(self, name=""):
        self.name = name
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.perf_counter() - self.start_time
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        
        log_str = f"[{self.name}] Latency: {latency:.4f}s | RAM: {ram_mb:.2f} MB"
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            log_str += f" | Peak VRAM: {vram_mb:.2f} MB"
        
        print(log_str)

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =========================
# Init Models
# =========================
print("Loading models...")
with PerfMonitor("Model Initialization"):
    df_model_path = "models/deepfilternet/DeepFilterNet3"
    df_model, df_state, _ = init_df(model_base_dir=df_model_path, post_filter=True)

    vad_model = torch.jit.load("models/silero/silero_vad.jit")
    vad_model.eval()
    vad_model = vad_model.to("cpu")

    asr_name = "models/phowhisper"
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=asr_name,
        tokenizer=asr_name,
        feature_extractor=asr_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=0 if device == "cuda" else -1,
    )

    from speechbrain.utils.fetching import LocalStrategy
    classifier_diar = EncoderClassifier.from_hparams(
        source="models/ecapa",
        savedir="models/ecapa",
        local_strategy=LocalStrategy.NO_LINK,
        run_opts={"device": device},
    )

    llm_name = "models/qwen"
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_name)
    model_llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
print("All models ready")

# =========================
# Functions
# =========================
def denoise_audio(input_file, output_file="clean.wav"):
    audio, sr = load_audio(input_file, sr=df_state.sr())
    enhanced = enhance(df_model, df_state, audio)
    save_audio(output_file, enhanced, df_state.sr())
    return output_file

def vad_split(wav_file):
    wav, sr = torchaudio.load(wav_file)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.squeeze()
    speech = get_speech_timestamps(
        wav.cpu(), vad_model, sampling_rate=16000,
        min_speech_duration_ms=200, min_silence_duration_ms=200, speech_pad_ms=50,
    )
    segments = [(s["start"] / 16000, s["end"] / 16000) for s in speech]
    return segments

def speech_to_text(wav_file, start, end):
    audio, sr = librosa.load(wav_file, sr=16000)
    audio = audio[int(start * sr): int(end * sr)]
    result = asr_pipeline(
        audio, generate_kwargs={"language": "vi", "task": "transcribe", "num_beams": 5, "temperature": 0},
    )
    text = result["text"]
    for s in ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]:
        text = text.replace(s, "")
    return text.strip()

def speaker_embedding(wav_file, start, end):
    audio, sr = torchaudio.load(wav_file)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio[:, int(start * 16000): int(end * 16000)]
    emb = classifier_diar.encode_batch(audio)
    return emb.squeeze()

def diarize(embeddings, speakers=2):
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)
    X = torch.stack(embeddings).cpu().numpy()
    speakers = min(speakers, len(embeddings))
    clustering = AgglomerativeClustering(n_clusters=speakers, metric="cosine", linkage="average")
    return clustering.fit_predict(X)

def correct_transcript(text):
    messages = [
        {"role": "system", "content": "Bạn là công cụ chuẩn hóa văn bản. Nhiệm vụ: Sửa lỗi chính tả, dấu câu cho văn bản tiếng Việt đầu vào, sửa các từ bị nghe nhầm (bị sai nghĩa do phát âm ngọng) để câu văn trở nên hợp lý theo ngữ cảnh hội thoại hỏi thăm giữa hai người sinh viên. Chỉ xuất ra văn bản đã sửa."},
        {"role": "user", "content": "sửa câu: cho mình hỏi bạn là sinh viên chiều nào vậy"},
        {"role": "assistant", "content": "Cho mình hỏi bạn là sinh viên trường nào vậy?"},
        {"role": "user", "content": f"Sửa câu sau:\nInput: hôm nay trời đẹp quá\nOutput: Hôm nay trời đẹp quá.\n\nSửa câu sau:\nInput: {text}\nOutput:"}
    ]
    prompt = tokenizer_llm.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer_llm(prompt, return_tensors="pt", add_special_tokens=False).to(model_llm.device)
    
    with torch.no_grad():
        output = model_llm.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    full_result = tokenizer_llm.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    final_text = full_result.split('\n')[0].replace("Output:", "").strip()
    return final_text

def normalize_speaker_labels(raw_speakers):
    chronological_speakers = []
    label_mapping = {}
    next_speaker_id = 1
    for original_label in raw_speakers:
        if original_label not in label_mapping:
            label_mapping[original_label] = next_speaker_id
            next_speaker_id += 1
        chronological_speakers.append(label_mapping[original_label])
    return chronological_speakers

# =========================
# Pipeline
# =========================
def run_pipeline(audio_file="test_audio.wav", speakers=2):
    print("\n" + "="*40)
    print("BẮT ĐẦU CHẠY PIPELINE")
    print("="*40)
    
    with PerfMonitor("TOTAL PIPELINE"):
        with PerfMonitor("1. Denoise Audio"):
            clean = denoise_audio(audio_file)
            
        with PerfMonitor("2. VAD Split"):
            segments = vad_split(clean)

        texts = []
        embs = []

        # Xử lý từng segment
        for i, (s, e) in enumerate(segments, start=1):
            print(f"\n--- Segment {i} ---")
            
            with PerfMonitor(f"  ASR"):
                raw_txt = speech_to_text(clean, s, e)
            print(f"  Raw ASR: {raw_txt}")

            with PerfMonitor(f"  LLM Correction"):
                corrected_txt = correct_transcript(raw_txt)
            
            with PerfMonitor(f"  Speaker Embedding"):
                emb = speaker_embedding(clean, s, e)

            texts.append(corrected_txt)
            embs.append(emb)

        print("\n--- Finalizing ---")
        with PerfMonitor("3. Diarization"):
            raw_speakers = diarize(embs, speakers=speakers)
            chronological_speakers = normalize_speaker_labels(raw_speakers)

        conversation = ""
        for i, t in enumerate(texts):
            conversation += f"Speaker {chronological_speakers[i]}: {t}\n"

        print("\n====== FINAL TEXT ======")
        print(conversation)
        
    return conversation

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "test_audio.wav"

    run_pipeline(audio_file)