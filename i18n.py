"""Lightweight, pure-Python translations for the Kivy mobile UI (zh / en).

No Qt, no widget binding: ``tr(key, **fmt)`` is a plain dictionary lookup.
The Kivy layer re-reads strings through ``App.tr`` (an ObservableProperty
trick lives in main.py) so switching language refreshes the UI.
"""
from __future__ import annotations

SUPPORTED_LANGUAGES = ("zh", "en")
DEFAULT_LANGUAGE = "zh"

current_language = DEFAULT_LANGUAGE

STRINGS: dict[str, dict[str, str]] = {
    "app_title": {"zh": "自动提取与歌词灌注", "en": "Auto Extract & Lyrics"},
    "section_audio": {"zh": "音频文件", "en": "Audio Files"},
    "pick_files": {"zh": "选择文件", "en": "Add Files"},
    "clear_files": {"zh": "清空", "en": "Clear"},
    "no_files": {"zh": "尚未选择音频文件", "en": "No audio files selected"},
    "files_added": {"zh": "已添加 {n} 个音频文件", "en": "{n} audio file(s) added"},
    "picker_unavailable": {
        "zh": "无法调用系统文件选择器: {err}",
        "en": "System file picker unavailable: {err}",
    },
    "ref_lyrics": {"zh": "参考歌词 (可选)", "en": "Reference Lyrics (Optional)"},
    "ref_lyrics_hint": {
        "zh": "输入确切歌词（纯文本）以提高对齐准确率",
        "en": "Enter exact lyrics (plain text) to improve alignment",
    },
    "section_options": {"zh": "处理选项", "en": "Options"},
    "target_lang": {"zh": "目标语言", "en": "Language"},
    "slicing_method": {"zh": "切片方法", "en": "Slicing"},
    "slice_smart": {"zh": "智能切片", "en": "Smart"},
    "slice_heuristic": {"zh": "启发式切片", "en": "Heuristic"},
    "slice_default": {"zh": "默认切片", "en": "Default"},
    "slice_grid": {"zh": "网格搜索切片", "en": "Grid Search"},
    "export_format": {"zh": "导出格式", "en": "Export"},
    "tempo_bpm": {"zh": "Tempo BPM", "en": "Tempo BPM"},
    "quant_step": {"zh": "MIDI 量化精度", "en": "Quantization"},
    "quant_off": {"zh": "不量化", "en": "Off"},
    "quant_1_4": {"zh": "1/4 音符", "en": "1/4 note"},
    "quant_1_8": {"zh": "1/8 音符", "en": "1/8 note"},
    "quant_1_16": {"zh": "1/16 音符", "en": "1/16 note"},
    "quant_1_32": {"zh": "1/32 音符", "en": "1/32 note"},
    "quant_1_64": {"zh": "1/64 音符", "en": "1/64 note"},
    "output_lyrics": {"zh": "输出歌词", "en": "Output Lyrics"},
    "save_dir": {"zh": "保存目录", "en": "Save Directory"},
    "section_models": {"zh": "模型路径", "en": "Model Paths"},
    "game_path": {"zh": "GAME 模型路径", "en": "GAME Model Path"},
    "tifa_path": {"zh": "TiFA 模型路径", "en": "TiFA Model Path"},
    "hfa_path": {"zh": "HubertFA 模型路径", "en": "HubertFA Model Path"},
    "asr_path": {"zh": "Qwen3-ASR 模型路径", "en": "Qwen3-ASR Model Path"},
    "phoneme_path": {"zh": "音素 ASR 模型路径", "en": "Phoneme ASR Model Path"},
    "pinyin_path": {"zh": "拼音 ASR 模型路径", "en": "Pinyin ASR Model Path"},
    "kashi_g2p_path": {"zh": "kashi-g2p 模型路径", "en": "kashi-g2p Model Path"},
    "rmvpe_path": {"zh": "RMVPE 模型路径", "en": "RMVPE Model Path"},
    "ui_language": {"zh": "界面语言", "en": "UI Language"},
    "run": {"zh": "开始提取", "en": "Start"},
    "stop": {"zh": "停止", "en": "Stop"},
    "stopping": {"zh": "正在停止...", "en": "Stopping..."},
    "idle": {"zh": "就绪", "en": "Ready"},
    "preparing": {"zh": "准备处理 {n} 个文件...", "en": "Preparing {n} file(s)..."},
    "processing": {"zh": "处理中 {i}/{n}: {f}", "en": "Processing {i}/{n}: {f}"},
    "log_title": {"zh": "运行日志", "en": "Log"},
    "err_no_audio": {"zh": "请至少选择一个音频文件", "en": "Please add at least one audio file"},
    "err_hybrid": {
        "zh": "错误: 推理管线未能加载: {err}",
        "en": "Error: inference pipeline failed to load: {err}",
    },
    "err_busy": {"zh": "任务正在运行", "en": "A task is already running"},
    "worker_processing": {"zh": "====== 正在处理: {f} ======", "en": "====== Processing: {f} ======"},
    "worker_success": {"zh": "提取成功！文件已保存至: {d}", "en": "Done! Files saved to: {d}"},
    "worker_cancelled": {"zh": "任务已被取消。", "en": "Task was cancelled."},
    "worker_stopped": {"zh": "任务已被停止。", "en": "Task was stopped."},
    "worker_error": {"zh": "发生错误:\n{tb}", "en": "An error occurred:\n{tb}"},
}

# (backend value, i18n key) option tables, shared by main.py / the .kv file.
TARGET_LANGUAGE_CHOICES = [("zh", "zh"), ("ja", "ja"), ("en", "en"), ("yue", "yue")]
SLICE_METHOD_CHOICES = [
    ("smart", "slice_smart"),
    ("heuristic", "slice_heuristic"),
    ("default", "slice_default"),
    ("grid", "slice_grid"),
]
EXPORT_FORMAT_CHOICES = [("mid", "MIDI"), ("ustx", "USTX"), ("vsqx", "VSQX")]
QUANT_STEP_CHOICES = [
    (0, "quant_off"),
    (480, "quant_1_4"),
    (240, "quant_1_8"),
    (120, "quant_1_16"),
    (60, "quant_1_32"),
    (30, "quant_1_64"),
]
DEFAULT_LYRIC_OUTPUT = {"zh": "hanzi", "ja": "romaji", "en": "word", "yue": "hanzi"}


def set_language(language: str) -> str:
    """Switch the active language; anything other than English maps to zh."""
    global current_language
    value = str(language or "").strip().lower()
    current_language = "en" if value in {"en", "english"} else "zh"
    return current_language


def get_language() -> str:
    return current_language


def tr(key: str, **kwargs) -> str:
    """Return the translated string for ``key`` (or ``key`` itself if unknown)."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = entry.get(current_language) or entry[DEFAULT_LANGUAGE]
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
