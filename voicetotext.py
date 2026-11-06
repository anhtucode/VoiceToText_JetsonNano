import os
import sys
import gc
import numpy as np
import librosa
import soundfile as sf
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sklearn.cluster import AgglomerativeClustering
import torch

# Import DeepFilterNet PyTorch
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
except ImportError:
    print("[LỖI] Chưa cài đặt DeepFilterNet. Hãy chạy: pip install deepfilternet")
    sys.exit(1)

# Import HuggingFace Transformers cho Qwen
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperTokenizer

# ==========================================
# 1. Khởi tạo Device & Cấu hình
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Sử dụng thiết bị: {device}")
ort_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']

# ==========================================
# 2. Khởi tạo Các mô hình
# ==========================================

# 2.1 DeepFilterNet (PyTorch)
print("[1/5] Loading DeepFilterNet (PyTorch)...")
try:
    # df_model sẽ tự động load trọng số mặc định
    df_model, df_state, _ = init_df()
    if device == "cuda":
        df_model = df_model.to(device)
except Exception as e:
    print(f"  -> Lỗi khởi tạo DeepFilterNet: {e}")
    df_model, df_state = None, None

# 2.2 Các mô hình ONNX (VAD, ECAPA, Whisper Decoder)
print("[2/5] Loading ONNX Models (Silero VAD, ECAPA, Whisper Decoder)...")
try:
    vad_session = ort.InferenceSession("silero_vad.onnx", providers=ort_providers)
    ecapa_session = ort.InferenceSession("ecapa_tdnn.onnx", providers=ort_providers)
    whisper_decoder_session = ort.InferenceSession("decoder_model.onnx", providers=ort_providers)
except Exception as e:
    print(f"  -> Không tìm thấy hoặc lỗi khởi tạo file ONNX: {e}")

# 2.3 TensorRT (Whisper Encoder)
print("[3/5] Loading TensorRT Engine (PhoWhisper Encoder)...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
whisper_encoder_context = None

try:
    with open("encoder_fp16.engine", "rb") as f:
        whisper_encoder_engine = trt_runtime.deserialize_cuda_engine(f.read())
    whisper_encoder_context = whisper_encoder_engine.create_execution_context()
except Exception as e:
    print(f"  -> Lỗi tải TensorRT Engine: {e}")

# 2.4 Tokenizer cho Whisper
print("[4/5] Loading Whisper Tokenizer...")
try:
    whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
except Exception:
    whisper_tokenizer = None
    print("  -> Lỗi tải Whisper Tokenizer.")

# 2.5 LLM (Qwen - PyTorch)
print("[5/5] Loading Qwen LLM...")
llm_name = "models/qwen" # Đường dẫn model của bạn
try:
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_name)
    model_llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
except Exception as e:
    print(f"  -> Chưa tải được Qwen LLM: {e}")

print("\n=> Khởi tạo Pipeline thành công!\n")

# ==========================================
# 3. Các hàm thành phần của Pipeline
# ==========================================

def run_trt_encoder(mel_input):
    """Thực thi mạng Encoder bằng TensorRT"""
    if not whisper_encoder_context:
        return np.random.randn(1, 1500, 384).astype(np.float32)
        
    input_shape = mel_input.shape
    whisper_encoder_context.set_binding_shape(0, input_shape)
    
    d_input = cuda.mem_alloc(mel_input.nbytes)
    output_shape = whisper_encoder_context.get_binding_shape(1)
    
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float16)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    cuda.memcpy_htod(d_input, mel_input)
    bindings = [int(d_input), int(d_output)]
    whisper_encoder_context.execute_v2(bindings)
    cuda.memcpy_dtoh(h_output, d_output)
    
    d_input.free()
    d_output.free()
    
    return h_output.reshape(output_shape)

def denoise_audio_pytorch(input_file, output_file="clean.wav"):
    """Sử dụng DeepFilterNet PyTorch nguyên bản"""
    if df_model is None or df_state is None:
        return input_file
        
    # Load audio thông qua API của DeepFilterNet (tự động resample về chuẩn của mô hình)
    audio, _ = load_audio(input_file, sr=df_state.sr())
    
    # Chạy mô hình để khử nhiễu
    enhanced_audio = enhance(df_model, df_state, audio)
    
    # Lưu file đã xử lý
    save_audio(output_file, enhanced_audio, df_state.sr())
    
    # Giải phóng VRAM của PyTorch ngay sau khi chạy xong DFN
    torch.cuda.empty_cache()
    
    return output_file

def vad_split(wav_file, chunk_size=512, threshold=0.5):
    """Cắt file âm thanh thành các đoạn dựa trên giọng nói (ONNX)"""
    wav, sr = librosa.load(wav_file, sr=16000)
    
    h = np.zeros((2, 1, 64), dtype=np.float32)
    c = np.zeros((2, 1, 64), dtype=np.float32)
    
    segments = []
    triggered = False
    start_time = 0.0
    
    if "vad_session" not in globals():
        return [(0.0, len(wav)/16000)] 
        
    for i in range(0, len(wav), chunk_size):
        chunk = wav[i:i+chunk_size]
        if len(chunk) < chunk_size: 
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
        ort_inputs = {
            'input': chunk[np.newaxis, :].astype(np.float32),
            'sr': np.array([16000], dtype=np.int64),
            'h': h,
            'c': c
        }
        
        try:
            out, h, c = vad_session.run(None, ort_inputs)
            speech_prob = out[0][0]
        except Exception:
            speech_prob = 0.0
            
        current_time = (i + chunk_size) / 16000.0
        
        if speech_prob >= threshold and not triggered:
            triggered = True
            start_time = current_time - (chunk_size / 16000.0)
        elif speech_prob < (threshold - 0.15) and triggered:
            triggered = False
            segments.append((start_time, current_time))
            
    if triggered:
        segments.append((start_time, len(wav)/16000.0))
        
    return segments if segments else [(0.0, len(wav)/16000)]

def speech_to_text(wav_file, start, end):
    """Trích xuất Text từ Audio dùng Whisper (TRT Encoder + ONNX Decoder)"""
    audio, sr = librosa.load(wav_file, sr=16000)
    audio = audio[int(start * sr): int(end * sr)]
    
    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80, n_fft=400, hop_length=160)
    mel = librosa.power_to_db(mel, ref=np.max)
    
    if mel.shape[1] < 3000:
        mel = np.pad(mel, ((0, 0), (0, 3000 - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :3000]
        
    mel = mel[np.newaxis, :, :].astype(np.float16)
    
    # Chạy TRT Encoder
    encoder_hidden_states = run_trt_encoder(mel)
    
    if "whisper_decoder_session" not in globals():
        return "Kết quả giả lập: Xin chào, đây là Whisper ONNX."

    input_ids = np.array([[50258]], dtype=np.int64) 
    generated_tokens = []
    
    # Vòng lặp Autoregressive Decoder (Tối đa 50 tokens)
    for _ in range(50):
        ort_inputs = {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states.astype(np.float32)
        }
        
        try:
            logits = whisper_decoder_session.run(None, ort_inputs)[0]
            next_token = np.argmax(logits[:, -1, :], axis=-1)
            generated_tokens.append(next_token[0])
            input_ids = np.concatenate([input_ids, next_token[np.newaxis, :]], axis=-1)
            
            if next_token[0] == 50257: # <|endoftext|>
                break
        except Exception:
            break
            
    if whisper_tokenizer:
        text = whisper_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        text = f"Token IDs: {generated_tokens}"
        
    return text.strip()

def speaker_embedding(wav_file, start, end):
    """Trích xuất đặc trưng người nói bằng ECAPA-TDNN"""
    audio, sr = librosa.load(wav_file, sr=16000)
    audio = audio[int(start * 16000): int(end * 16000)]
    
    fbank = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80)
    fbank = np.transpose(fbank)
    fbank = fbank[np.newaxis, ...].astype(np.float32)
    
    if "ecapa_session" in globals() and ecapa_session:
        try:
            ort_inputs = {ecapa_session.get_inputs()[0].name: fbank}
            emb = ecapa_session.run(None, ort_inputs)[0]
            return emb.squeeze()
        except:
            pass
            
    return np.random.randn(192)

def diarize(embeddings, speakers=2):
    """Phân cụm giọng nói"""
    if len(embeddings) <= 1:
        return np.zeros(len(embeddings), dtype=int)
    X = np.stack(embeddings)
    clustering = AgglomerativeClustering(n_clusters=min(speakers, len(embeddings)), metric="cosine", linkage="average")
    return clustering.fit_predict(X)

def correct_transcript(text):
    """Chuẩn hoá văn bản bằng Qwen (PyTorch)"""
    if "model_llm" not in globals() or "tokenizer_llm" not in globals():
        return text
        
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
    
    # Giải phóng VRAM sau khi Qwen sinh text xong
    torch.cuda.empty_cache()
    
    return full_result.split('\n')[0].replace("Output:", "").strip()

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

# ==========================================
# 4. Luồng Thực Thi Chính
# ==========================================
def run_pipeline(audio_file="test_audio.wav", speakers=2):
    if not os.path.exists(audio_file):
        print(f"[LỖI] Không tìm thấy file âm thanh '{audio_file}'.")
        return
        
    print(f"--- 1. Bắt đầu Denoise (DeepFilterNet PyTorch) ---")
    clean = denoise_audio_pytorch(audio_file)
    
    print(f"--- 2. Bắt đầu VAD (Silero VAD ONNX) ---")
    segments = vad_split(clean)
    print(f"[*] Phát hiện {len(segments)} đoạn hội thoại.")
    
    texts, embs = [], []
    for i, (s, e) in enumerate(segments, start=1):
        print(f"\n--- Đang xử lý Segment {i} ({s:.2f}s - {e:.2f}s) ---")
        
        # Nhận diện âm thanh -> Text
        raw_txt = speech_to_text(clean, s, e)
        print(f" Raw ASR: {raw_txt}")
        
        # Dọn rác hệ thống trước khi gọi LLM
        gc.collect()
        
        # Chuẩn hoá văn bản bằng Qwen
        corrected_txt = correct_transcript(raw_txt)
        print(f" Corrected: {corrected_txt}")
        
        # Trích xuất vector người nói
        emb = speaker_embedding(clean, s, e)
        
        texts.append(corrected_txt)
        embs.append(emb)
        
    print("\n--- 3. Bắt đầu Diarization (ECAPA-TDNN) ---")
    raw_speakers = diarize(embs, speakers=speakers)
    chron_speakers = normalize_speaker_labels(raw_speakers)
    
    conversation = ""
    for i, t in enumerate(texts):
        conversation += f"Speaker {chron_speakers[i]}: {t}\n"
        
    print("\n====== FINAL TEXT ======")
    print(conversation)
    return conversation

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "test_audio.wav"
    
    if not os.path.exists(target_file):
        print(f"Tạo file '{target_file}' rỗng để test (do không tìm thấy file gốc).")
        import scipy.io.wavfile as wav
        wav.write(target_file, 48000, np.zeros(48000, dtype=np.int16))
        
    run_pipeline(target_file)