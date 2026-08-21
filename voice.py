# 语音输入输出客户端：百炼 OpenAI 兼容端点（Paraformer ASR + CosyVoice TTS）
# 音频全程内存处理不落盘；错误处理与重试策略对齐 llm/llm.py
import os
import re
import time

from observability.events import event_logger
from observability.logger import get_logger

_logger = get_logger("voice")

try:
    import requests
except ImportError:
    requests = None


class VoiceError(RuntimeError):
    """语音服务调用错误，消息可直接展示给用户。"""


DEFAULT_VOICE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ASR_MODEL = "paraformer-v2"
DEFAULT_TTS_MODEL = "cosyvoice-v1"
DEFAULT_TTS_VOICE = "longwan"
# CosyVoice 单次 input 上限约 500 字符，留余量按 400 分段合成后拼接（mp3 帧可直接串联播放）
TTS_CHUNK_LIMIT = 400


def is_configured():
    """DASHSCOPE_API_KEY 非空即视为语音功能就绪。"""
    return bool(os.getenv("DASHSCOPE_API_KEY", "").strip())


def test_connection(api_key, base_url=None, tts_voice=None):
    """连通性探针：用 4 字短文本做一次真实合成（成本可忽略），返回 (ok, message)。

    能同时验证 key 有效性、端点可达与音色可用；ASR 与 TTS 共用同一 key
    与鉴权方式，合成通过即代表语音链路可用。
    """
    base = (base_url or "").strip().rstrip("/") or DEFAULT_VOICE_URL
    voice_name = (tts_voice or "").strip() or DEFAULT_TTS_VOICE
    started = time.perf_counter()
    try:
        response = _post(
            f"{base}/audio/speech",
            api_key,
            json={
                "model": DEFAULT_TTS_MODEL,
                "voice": voice_name,
                "input": "连接测试",
                "response_format": "mp3",
            },
        )
        audio = response.content or b""
        elapsed = _elapsed_ms(started)
        if not audio:
            event_logger.emit("voice_call", kind="tts_test", model=DEFAULT_TTS_MODEL, chars=4, elapsed_ms=elapsed, error=True, message="empty audio")
            return False, "语音服务返回空音频，请检查音色设置"
        event_logger.emit("voice_call", kind="tts_test", model=DEFAULT_TTS_MODEL, chars=4, elapsed_ms=elapsed, error=False)
        return True, f"语音服务连接正常（往返 {elapsed:.0f}ms，返回 {len(audio)} 字节音频）"
    except VoiceError as exc:
        event_logger.emit("voice_call", kind="tts_test", model=DEFAULT_TTS_MODEL, chars=4, elapsed_ms=_elapsed_ms(started), error=True, message=str(exc))
        return False, str(exc)


def _post(url, api_key, **kwargs):
    """与 llm.py 一致：timeout=30，网络异常/5xx 重试一次，4xx 直接抛错。"""
    if requests is None:
        raise VoiceError("缺少 requests 依赖，无法调用语音服务")

    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {api_key}"

    response = None
    for attempt in range(2):
        try:
            response = requests.post(url, headers=headers, timeout=30, **kwargs)
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise VoiceError(f"语音服务网络异常: {exc}")
        if response.status_code < 500 or attempt == 1:
            break
        _logger.info("voice service returned %s, retrying once", response.status_code)
        time.sleep(0.5)

    if response.status_code >= 400:
        message = ""
        try:
            data = response.json()
            if isinstance(data, dict):
                message = (data.get("error") or {}).get("message", "")
        except ValueError:
            pass
        detail = message or (response.text or "")[:200]
        raise VoiceError(f"语音服务返回 {response.status_code}: {detail}")
    return response


def transcribe(audio, filename="recording.webm", mime="audio/webm"):
    """语音转文字（Paraformer），返回识别文本。"""
    base = os.getenv("DASHSCOPE_VOICE_URL", DEFAULT_VOICE_URL).rstrip("/")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    model = os.getenv("DASHSCOPE_ASR_MODEL", DEFAULT_ASR_MODEL)

    started = time.perf_counter()
    try:
        response = _post(
            f"{base}/audio/transcriptions",
            api_key,
            files={"file": (filename, audio, mime)},
            data={"model": model},
        )
        data = response.json()
        text = (data.get("text") or "").strip() if isinstance(data, dict) else ""
        if not text:
            raise VoiceError("未识别到语音内容，请重新录音")
        event_logger.emit("voice_call", kind="asr", model=model, elapsed_ms=_elapsed_ms(started), error=False)
        return text
    except VoiceError as exc:
        event_logger.emit("voice_call", kind="asr", model=model, elapsed_ms=_elapsed_ms(started), error=True, message=str(exc))
        raise


def synthesize(text):
    """文字转语音（CosyVoice），返回 mp3 音频字节。

    长文本按句边界分段（单段不超过 CosyVoice 输入上限），
    逐段合成后直接拼接 mp3 字节——mp3 帧相互独立，串联仍可顺序播放。
    """
    chunks = _split_for_tts(text)
    parts = []
    for chunk in chunks:
        parts.append(_synthesize_chunk(chunk))
    return b"".join(parts)


def _synthesize_chunk(text):
    base = os.getenv("DASHSCOPE_VOICE_URL", DEFAULT_VOICE_URL).rstrip("/")
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()

    started = time.perf_counter()
    try:
        response = _post(
            f"{base}/audio/speech",
            api_key,
            json={
                "model": DEFAULT_TTS_MODEL,
                "voice": os.getenv("DASHSCOPE_TTS_VOICE", DEFAULT_TTS_VOICE),
                "input": text,
                "response_format": "mp3",
            },
        )
        audio = response.content
        if not audio:
            raise VoiceError("语音合成返回空音频")
        event_logger.emit("voice_call", kind="tts", model=DEFAULT_TTS_MODEL, chars=len(text), elapsed_ms=_elapsed_ms(started), error=False)
        return audio
    except VoiceError as exc:
        event_logger.emit("voice_call", kind="tts", model=DEFAULT_TTS_MODEL, chars=len(text), elapsed_ms=_elapsed_ms(started), error=True, message=str(exc))
        raise


def _split_for_tts(text):
    """按句末标点切分并合并为不超过 TTS_CHUNK_LIMIT 的段，无标点时硬切。"""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= TTS_CHUNK_LIMIT:
        return [text]
    sentences = [part for part in re.split(r"(?<=[。！？!?；;\n])", text) if part.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        while len(sentence) > TTS_CHUNK_LIMIT:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sentence[:TTS_CHUNK_LIMIT])
            sentence = sentence[TTS_CHUNK_LIMIT:]
        if len(current) + len(sentence) > TTS_CHUNK_LIMIT:
            chunks.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        chunks.append(current)
    return chunks


def _elapsed_ms(started):
    return round((time.perf_counter() - started) * 1000, 1)
