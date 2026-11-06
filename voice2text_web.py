#!/usr/bin/env python3
"""Web UI backend for voice2text_nollm.py using only the Python standard library."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
INDEX_FILE = PROJECT_DIR / "voice2text_web.html"
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MIN_SPEAKERS = 1
MAX_SPEAKERS = 10
ASR_MODELS = {
    "phowhisper": {
        "label": "PhoWhisper",
        "path": "models/phowhisper",
    },
    "phowhisper_finetune": {
        "label": "PhoWhisper Fine-tune",
        "path": "models/phowhisper_finetune",
    },
    "phowhisper_origin": {
        "label": "PhoWhisper Origin",
        "path": "models/phowhisper-origin",
    },
}

_model_module: Any | None = None
_model_error: str | None = None
_active_asr_model: str | None = None
_active_use_llm: bool | None = None
_model_lock = threading.Lock()
_pipeline_lock = threading.Lock()
_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_active_job_id: str | None = None
MAX_FINISHED_JOBS = 20


class JobBusyError(RuntimeError):
    pass


def load_model(asr_model: str, use_llm: bool) -> Any:
    """Initialize the pipeline and apply the selected ASR/LLM configuration."""
    global _model_module, _model_error, _active_asr_model, _active_use_llm

    if asr_model not in ASR_MODELS:
        raise ValueError("ASR model không hợp lệ.")

    model_path = ASR_MODELS[asr_model]["path"]
    if not (PROJECT_DIR / model_path).is_dir():
        raise NotADirectoryError(f"Không tìm thấy {model_path}")

    with _model_lock:
        if (
            _model_module is not None
            and _active_asr_model == asr_model
            and _active_use_llm == use_llm
        ):
            return _model_module

        try:
            os.chdir(PROJECT_DIR)
            if _model_module is None:
                os.environ["VOICE2TEXT_ASR_MODEL"] = model_path
                _model_module = importlib.import_module("voice2text_nollm")
            elif _active_asr_model != asr_model:
                _active_asr_model = None
                _model_module.load_asr_model(model_path)

            if use_llm:
                _model_module.load_llm_model("models/qwen")
            else:
                _model_module.unload_llm_model()

            _active_asr_model = asr_model
            _active_use_llm = use_llm
            _model_error = None
            return _model_module
        except Exception as exc:
            _active_asr_model = None
            _active_use_llm = None
            _model_error = f"{type(exc).__name__}: {exc}"
            raise


def model_status() -> dict[str, Any]:
    with _jobs_lock:
        active_job_id = _active_job_id
    return {
        "ready": (
            _model_module is not None
            and _active_asr_model is not None
            and _active_use_llm is not None
        ),
        "error": _model_error,
        "busy": _pipeline_lock.locked() or active_job_id is not None,
        "active_job_id": active_job_id,
        "active_asr_model": _active_asr_model,
        "active_use_llm": _active_use_llm,
        "asr_models": ASR_MODELS,
    }


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)
            _jobs[job_id]["updated_at"] = time.time()


def _prune_finished_jobs() -> None:
    finished = [
        job for job in _jobs.values()
        if job["status"] in {"completed", "failed"}
    ]
    finished.sort(key=lambda job: job.get("completed_at", 0), reverse=True)
    for job in finished[MAX_FINISHED_JOBS:]:
        _jobs.pop(job["job_id"], None)


def run_transcription_job(
    job_id: str,
    audio_data: bytes,
    content_type: str,
    speakers: int,
    use_llm: bool,
) -> None:
    global _active_job_id

    update_job(
        job_id,
        status="processing",
        progress="Đang chuyển đổi và xử lý audio...",
        started_at=time.time(),
    )
    try:
        suffix = ".ogg" if "ogg" in content_type else ".webm"
        with _pipeline_lock, tempfile.TemporaryDirectory(prefix="voice2text_web_") as temp_dir:
            source = Path(temp_dir) / f"recording{suffix}"
            wav_file = Path(temp_dir) / "recording_16k.wav"
            source.write_bytes(audio_data)
            convert_to_wav(source, wav_file)
            result = _model_module.run_pipeline(
                str(wav_file),
                speakers=speakers,
                use_llm=use_llm,
            )

        update_job(
            job_id,
            status="completed",
            progress="Hoàn tất",
            result=result.strip(),
            completed_at=time.time(),
        )
    except subprocess.TimeoutExpired:
        update_job(
            job_id,
            status="failed",
            progress="Xử lý thất bại",
            error="Chuyển đổi audio quá thời gian cho phép.",
            completed_at=time.time(),
        )
    except Exception as exc:
        update_job(
            job_id,
            status="failed",
            progress="Xử lý thất bại",
            error=f"{type(exc).__name__}: {exc}",
            completed_at=time.time(),
        )
    finally:
        with _jobs_lock:
            if _active_job_id == job_id:
                _active_job_id = None


def create_transcription_job(
    audio_data: bytes,
    content_type: str,
    speakers: int,
    use_llm: bool,
) -> dict[str, Any]:
    global _active_job_id

    now = time.time()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        if _active_job_id is not None:
            raise JobBusyError("Pipeline đang xử lý một bản ghi khác.")
        _prune_finished_jobs()
        job = {
            "job_id": job_id,
            "status": "queued",
            "progress": "Đang chờ xử lý...",
            "result": None,
            "error": None,
            "speakers_requested": speakers,
            "use_llm": use_llm,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
        _jobs[job_id] = job
        _active_job_id = job_id

    worker = threading.Thread(
        target=run_transcription_job,
        args=(job_id, audio_data, content_type, speakers, use_llm),
        name=f"voice2text-job-{job_id[:8]}",
        daemon=True,
    )
    worker.start()
    return dict(job)


def convert_to_wav(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        detail = result.stderr.strip() or "ffmpeg không thể chuyển đổi bản ghi âm."
        raise RuntimeError(detail)


class VoiceHandler(BaseHTTPRequestHandler):
    server_version = "Voice2TextWeb/1.0"

    def _send_bytes(self, payload: bytes, content_type: str, status: int = 200) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(payload, "application/json; charset=utf-8", status)

    def _read_audio_body(self) -> bytes:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Thiếu Content-Length.")

        size = int(content_length)
        if size <= 0:
            raise ValueError("Bản ghi âm đang trống.")
        if size > MAX_AUDIO_BYTES:
            raise ValueError("Bản ghi âm vượt quá giới hạn 100 MB.")
        return self.rfile.read(size)

    def _read_json_body(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > 10_000:
            raise ValueError("JSON request không hợp lệ.")
        return json.loads(self.rfile.read(size).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            try:
                self._send_bytes(INDEX_FILE.read_bytes(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send_json({"ok": False, "error": f"Không tìm thấy {INDEX_FILE.name}"}, 500)
            return

        if self.path == "/api/status":
            self._send_json({"ok": True, **model_status()})
            return

        if self.path.startswith("/api/jobs/"):
            job_id = self.path.removeprefix("/api/jobs/").strip("/")
            job = get_job(job_id)
            if job is None:
                self._send_json({"ok": False, "error": "Không tìm thấy job."}, 404)
            else:
                self._send_json({"ok": True, **job})
            return

        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        self._send_json({"ok": False, "error": "Không tìm thấy endpoint."}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/model/start":
            try:
                request_data = self._read_json_body()
                asr_model = request_data.get("asr_model", "")
                use_llm = request_data.get("use_llm", False)
                if not isinstance(use_llm, bool):
                    raise ValueError("Tùy chọn LLM không hợp lệ.")
                with _jobs_lock:
                    active_job_id = _active_job_id
                if _pipeline_lock.locked() or active_job_id is not None:
                    self._send_json({"ok": False, "error": "Pipeline đang bận."}, 409)
                    return

                with _pipeline_lock:
                    load_model(asr_model, use_llm)

                mode_label = "có LLM" if use_llm else "không LLM"
                self._send_json({
                    "ok": True,
                    "ready": True,
                    "active_asr_model": asr_model,
                    "active_use_llm": use_llm,
                    "message": f"{ASR_MODELS[asr_model]['label']} ({mode_label}) đã sẵn sàng.",
                })
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "ready": False, "error": str(exc)}, 400)
            except Exception as exc:
                self._send_json(
                    {"ok": False, "ready": False, "error": f"{type(exc).__name__}: {exc}"},
                    500,
                )
            return

        if self.path == "/api/transcribe":
            if (
                _model_module is None
                or _active_asr_model is None
                or _active_use_llm is None
            ):
                self._send_json({"ok": False, "error": "Hãy khởi động model trước."}, 409)
                return

            if _pipeline_lock.locked():
                self._send_json({"ok": False, "error": "Pipeline đang xử lý một bản ghi khác."}, 409)
                return

            try:
                try:
                    speakers = int(self.headers.get("X-Speaker-Count", "2"))
                except ValueError as exc:
                    raise ValueError("Số speaker phải là số nguyên.") from exc
                if not MIN_SPEAKERS <= speakers <= MAX_SPEAKERS:
                    raise ValueError(
                        f"Số speaker phải nằm trong khoảng {MIN_SPEAKERS}–{MAX_SPEAKERS}."
                    )

                audio_data = self._read_audio_body()
                content_type = self.headers.get("Content-Type", "audio/webm").lower()
                job = create_transcription_job(
                    audio_data,
                    content_type,
                    speakers,
                    _active_use_llm,
                )

                self._send_json({
                    "ok": True,
                    "job_id": job["job_id"],
                    "status": job["status"],
                    "message": "Audio đã được nhận và đang xử lý nền.",
                }, 202)
            except JobBusyError as exc:
                self._send_json({"ok": False, "error": str(exc)}, 409)
            except (ValueError, RuntimeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)
            return

        self._send_json({"ok": False, "error": "Không tìm thấy endpoint."}, 404)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI cho voice2text_nollm.py")
    parser.add_argument("--host", default="127.0.0.1", help="Địa chỉ bind (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Cổng web (mặc định: 8000)")
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)
    server = ThreadingHTTPServer((args.host, args.port), VoiceHandler)
    print(f"Voice2Text Web đang chạy tại http://{args.host}:{args.port}")
    print("Nhấn Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐang dừng server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
