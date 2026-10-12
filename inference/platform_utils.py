"""Runtime platform detection and the Android ASR downgrade policy.

Android cannot load the Windows llama.cpp DLLs that Qwen3-ASR needs, so on
Android the pipeline is forced onto the pure-ONNX phoneme ASR engines
(PinyinASR for Chinese, RomajiASR for Japanese).
"""
from __future__ import annotations

import os
import sys

# Languages that have a pure-ONNX ASR engine usable without Qwen3-ASR.
ANDROID_SUPPORTED_LANGUAGES = ("zh", "ja")


def is_android() -> bool:
    """True when running under python-for-android / Kivy on Android."""
    if hasattr(sys, "getandroidapilevel"):
        return True
    try:
        from kivy.utils import platform

        if platform == "android":
            return True
    except Exception:
        pass
    # Fallbacks when Kivy is not imported yet (p4a always sets these).
    return "ANDROID_ARGUMENT" in os.environ or (
        "ANDROID_ROOT" in os.environ and "ANDROID_DATA" in os.environ
    )


def qwen_asr_available() -> bool:
    """Qwen3-ASR (llama.cpp DLL) is Windows/desktop only."""
    return not is_android()


def apply_android_asr_policy(cfg) -> list[str]:
    """Downgrade ``cfg`` in place to ONNX-only ASR engines on Android.

    Returns human-readable notes describing what was changed (empty when
    nothing changed / not on Android). Raises ``ValueError`` for languages
    that have no on-device engine (en / yue need Qwen text ASR).
    """
    if not is_android():
        return []
    notes: list[str] = []
    language = str(getattr(cfg, "language", "") or "").strip().lower()
    if language in {"zh", "zh-pinyin"}:
        if str(cfg.chinese_asr_engine or "").strip().lower() != "pinyin":
            notes.append("Android: Qwen3-ASR unavailable, using PinyinASR for Chinese.")
        cfg.chinese_asr_engine = "pinyin"
        cfg.language = "zh"
        cfg.lyric_output_mode = "pinyin"  # PinyinASR can only emit pinyin lyrics
    elif language == "ja":
        if str(cfg.japanese_asr_engine or "").strip().lower() != "romaji":
            notes.append("Android: Qwen3-ASR unavailable, using RomajiASR for Japanese.")
        cfg.japanese_asr_engine = "romaji"
        if cfg.lyric_output_mode not in {"romaji", "kana"}:
            cfg.lyric_output_mode = "romaji"
    else:
        raise ValueError(
            f"Language '{language}' requires Qwen3-ASR, which is not available on "
            f"Android. Supported on Android: {', '.join(ANDROID_SUPPORTED_LANGUAGES)}."
        )
    return notes
