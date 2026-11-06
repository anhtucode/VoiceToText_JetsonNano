import os
import sys
import time
import gc
import psutil
import torch
import torchaudio
import librosa
import numpy as np
import warnings
import logging

# 1. TẮT CÁC CẢNH BÁO RÁC TỪ THƯ VIỆN
warnings.filterwarnings("ignore")
import transformers
transformers.logging.set_verbosity_error()

# 2. CẤU HÌNH LOGGING ĐỂ LƯU RA FILE VÀ IN RA MÀN HÌNH
def setup_logger(log_file="pipeline.log"):
    logger = logging.getLogger("VoicePipeline")
    logger.setLevel(logging.INFO)
    
    # Xóa các handler cũ nếu chạy trong notebook/môi trường giữ state
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler ghi vào file
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)

    # Handler in ra màn hình (Console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s') # In ra console gọn gàng, không cần thời gian
    console_handler.setFormatter(console_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger("performance_and_result.log")

from df.enhance import enhance, init_df, load_audio, save_audio
from pathlib import Path
from utils_vad import get_speech_timestamps
from transformers import pipeline
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering

# =========================
# Công cụ đo lường Performance
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
        
        logger.info(log_str)

# =========================
# Init Models
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
logger.info("Loading models...")

asr_name = None
asr_pipeline = None
llm_name = None
tokenizer_llm = None
model_llm = None

def load_asr_model(model_path):
    """Nạp hoặc chuyển ASR model mà không cần nạp lại các model còn lại."""
    global asr_name, asr_pipeline

    model_path = str(model_path)
    if not Path(model_path).is_dir():
        raise NotADirectoryError(f"ASR model directory not found at {model_path}")
    if asr_pipeline is not None and asr_name == model_path:
        return asr_name

    logger.info(f"Loading ASR model: {model_path}")
    asr_pipeline = None
    asr_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loaded_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        tokenizer=model_path,
        feature_extractor=model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=0 if device == "cuda" else -1,
    )
    asr_pipeline = loaded_pipeline
    asr_name = model_path
    logger.info(f"ASR model ready: {model_path}")
    return asr_name

def unload_llm_model():
    """Giải phóng Qwen khi chuyển về chế độ không dùng LLM."""
    global llm_name, tokenizer_llm, model_llm

    if model_llm is None and tokenizer_llm is None:
        return
    logger.info("Unloading LLM model")
    model_llm = None
    tokenizer_llm = None
    llm_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("LLM model unloaded")

def load_llm_model(model_path="models/qwen"):
    """Nạp Qwen theo yêu cầu; không chiếm RAM khi chế độ LLM đang tắt."""
    global llm_name, tokenizer_llm, model_llm

    model_path = str(model_path)
    if not Path(model_path).is_dir():
        raise NotADirectoryError(f"LLM model directory not found at {model_path}")
    if model_llm is not None and tokenizer_llm is not None and llm_name == model_path:
        return llm_name

    unload_llm_model()
    logger.info(f"Loading LLM model: {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer_llm = loaded_tokenizer
    model_llm = loaded_model
    llm_name = model_path
    logger.info(f"LLM model ready: {model_path}")
    return llm_name

with PerfMonitor("Model Initialization"):
    df_model_path = "models/deepfilternet/DeepFilterNet3"
    df_model, df_state, _ = init_df(model_base_dir=df_model_path, post_filter=True)

    vad_model = torch.jit.load("models/silero/silero_vad.jit")
    vad_model.eval()
    vad_model = vad_model.to("cpu")

    load_asr_model(os.environ.get("VOICE2TEXT_ASR_MODEL", "models/phowhisper"))

    from speechbrain.utils.fetching import LocalStrategy
    classifier_diar = EncoderClassifier.from_hparams(
        source="models/ecapa",
        savedir="models/ecapa",
        local_strategy=LocalStrategy.NO_LINK,
        run_opts={"device": device},
    )

logger.info("All models ready")

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
        audio, generate_kwargs={"language": "vi", "task": "transcribe", "num_beams": 5},
    )
    text = result["text"]
    for s in ["<|startoftranscript|>", "<|vi|>", "<|transcribe|>", "<|notimestamps|>"]:
        text = text.replace(s, "")
    return text.strip()

def correct_transcript(text):
    """Sửa chính tả và dấu câu bằng Qwen, theo logic của voice2text_llm.py."""
    if tokenizer_llm is None or model_llm is None:
        raise RuntimeError("LLM chưa được khởi động.")

    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là công cụ chuẩn hóa văn bản. Nhiệm vụ: Sửa lỗi chính tả, "
                "dấu câu cho văn bản tiếng Việt đầu vào. Chỉ xuất ra văn bản đã "
                "sửa, tuyệt đối không giải thích."
            ),
        },
        {
            "role": "user",
            "content": "sửa câu: cho mình hỏi bạn là sinh viên chiều nào vậy",
        },
        {
            "role": "assistant",
            "content": "Cho mình hỏi bạn là sinh viên trường nào vậy?",
        },
        {
            "role": "user",
            "content": f"Sửa câu sau:\nInput: {text}\nOutput:",
        },
    ]
    prompt = tokenizer_llm.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer_llm(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model_llm.device)

    with torch.no_grad():
        output = model_llm.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            top_p=None,
            temperature=None,
        )

    full_result = tokenizer_llm.decode(
        output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    final_text = full_result.split("\n")[0]
    for junk in ["Output:", "Input:", "Sửa câu sau:", '"']:
        final_text = final_text.replace(junk, "")
    return final_text.strip()

def speaker_embedding(wav_file, start, end):
    audio, sr = torchaudio.load(wav_file)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    audio = audio[:, int(start * 16000): int(end * 16000)]
    emb = classifier_diar.encode_batch(audio)
    return emb.squeeze()

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

def diarize(embeddings, speakers=2):
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)
    X = torch.stack(embeddings).cpu().numpy()
    speakers = min(speakers, len(embeddings))
    clustering = AgglomerativeClustering(n_clusters=speakers, metric="cosine", linkage="average")
    return clustering.fit_predict(X)

# =========================
# Pipeline
# =========================
def run_pipeline(audio_file="test_audio.wav", speakers=2, use_llm=False):
    logger.info("\n" + "="*40)
    logger.info("BẮT ĐẦU CHẠY PIPELINE")
    logger.info("="*40)
    
    with PerfMonitor("TOTAL PIPELINE"):
        with PerfMonitor("1. Denoise Audio"):
            clean = denoise_audio(audio_file)
            
        with PerfMonitor("2. VAD Split"):
            segments = vad_split(clean)

        texts = []
        embs = []

        for i, (s, e) in enumerate(segments, start=1):
            logger.info(f"\n--- Segment {i} ---")
            
            with PerfMonitor("  ASR"):
                raw_txt = speech_to_text(clean, s, e)
            logger.info(f"  ASR Text: {raw_txt}")

            final_txt = raw_txt
            if use_llm:
                with PerfMonitor("  LLM Correction"):
                    final_txt = correct_transcript(raw_txt)
                logger.info(f"  LLM Text: {final_txt}")
            
            with PerfMonitor("  Speaker Embedding"):
                emb = speaker_embedding(clean, s, e)

            texts.append(final_txt)
            embs.append(emb)

        logger.info("\n--- Finalizing ---")
        with PerfMonitor("3. Diarization"):
            raw_speakers = diarize(embs, speakers=speakers)
            chronological_speakers = normalize_speaker_labels(raw_speakers)

        conversation = ""
        for i, t in enumerate(texts):
            conversation += f"Speaker {chronological_speakers[i]}: {t}\n"

        logger.info("\n====== FINAL TEXT ======")
        logger.info(f"\n{conversation}")
        
    return conversation

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "test_audio.wav"

    run_pipeline(audio_file)
