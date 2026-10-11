"""Lightweight, pure-Python translations for the Kivy mobile UI.

Supported languages:
  - zh_tw (繁體中文, 預設 Default)
  - ja (日本語)
  - en (English)
  - zh_cn (简体中文)

No Qt, no widget binding: ``tr(key, **fmt)`` is a plain dictionary lookup.
The Kivy layer re-reads strings through ``App.tr`` so switching language refreshes the UI.
"""
from __future__ import annotations

SUPPORTED_LANGUAGES = ("zh_tw", "ja", "en", "zh_cn")
DEFAULT_LANGUAGE = "zh_tw"

UI_LANG_CHOICES = [
    ("zh_tw", "繁體中文"),
    ("ja", "日本語"),
    ("en", "English"),
    ("zh_cn", "简体中文"),
]

current_language = DEFAULT_LANGUAGE

STRINGS: dict[str, dict[str, str]] = {
    "app_title": {
        "zh_tw": "自動音訊轉 MIDI 與歌詞對齊",
        "zh_cn": "自动提取与歌词灌注",
        "en": "Auto Extract & Lyrics",
        "ja": "音声 MIDI 変換＆歌詞アライメント",
    },
    "section_audio": {
        "zh_tw": "音訊檔案",
        "zh_cn": "音频文件",
        "en": "Audio Files",
        "ja": "音声ファイル",
    },
    "pick_files": {
        "zh_tw": "選擇檔案",
        "zh_cn": "选择文件",
        "en": "Add Files",
        "ja": "ファイル追加",
    },
    "clear_files": {
        "zh_tw": "清空",
        "zh_cn": "清空",
        "en": "Clear",
        "ja": "クリア",
    },
    "no_files": {
        "zh_tw": "尚未選擇音訊檔案",
        "zh_cn": "尚未选择音频文件",
        "en": "No audio files selected",
        "ja": "音声ファイルが未選択です",
    },
    "files_added": {
        "zh_tw": "已加入 {n} 個音訊檔案",
        "zh_cn": "已添加 {n} 个音频文件",
        "en": "{n} audio file(s) added",
        "ja": "{n} 個の音声ファイルを追加しました",
    },
    "picker_unavailable": {
        "zh_tw": "無法開啟系統檔案選擇器: {err}",
        "zh_cn": "无法调用系统文件选择器: {err}",
        "en": "System file picker unavailable: {err}",
        "ja": "ファイルピッカーを開けませんでした: {err}",
    },
    "ref_lyrics": {
        "zh_tw": "參考歌詞 (選填)",
        "zh_cn": "参考歌词 (可选)",
        "en": "Reference Lyrics (Optional)",
        "ja": "参考歌詞 (任意)",
    },
    "ref_lyrics_hint": {
        "zh_tw": "輸入確切歌詞（純文字）以提升對齊準確率",
        "zh_cn": "输入确切歌词（纯文本）以提高对齐准确率",
        "en": "Enter exact lyrics (plain text) to improve alignment",
        "ja": "歌詞テキストを入力するとアライメント精度が向上します",
    },
    "section_options": {
        "zh_tw": "處理選項",
        "zh_cn": "处理选项",
        "en": "Options",
        "ja": "処理設定",
    },
    "target_lang": {
        "zh_tw": "目標語言",
        "zh_cn": "目标语言",
        "en": "Language",
        "ja": "対象言語",
    },
    "lang_zh": {
        "zh_tw": "中文 (普通話/國語)",
        "zh_cn": "中文 (普通话)",
        "en": "Chinese (Mandarin)",
        "ja": "中国語 (標準語)",
    },
    "lang_ja": {
        "zh_tw": "日語",
        "zh_cn": "日语",
        "en": "Japanese",
        "ja": "日本語",
    },
    "lang_en": {
        "zh_tw": "英語",
        "zh_cn": "英语",
        "en": "English",
        "ja": "英語",
    },
    "lang_yue": {
        "zh_tw": "粵語",
        "zh_cn": "粤语",
        "en": "Cantonese",
        "ja": "広東語",
    },
    "slicing_method": {
        "zh_tw": "切片方法",
        "zh_cn": "切片方法",
        "en": "Slicing",
        "ja": "スライス方式",
    },
    "slice_smart": {
        "zh_tw": "智慧切片",
        "zh_cn": "智能切片",
        "en": "Smart",
        "ja": "スマート",
    },
    "slice_heuristic": {
        "zh_tw": "啟發式切片",
        "zh_cn": "启发式切片",
        "en": "Heuristic",
        "ja": "ヒューリスティック",
    },
    "slice_default": {
        "zh_tw": "預設切片",
        "zh_cn": "默认切片",
        "en": "Default",
        "ja": "デフォルト",
    },
    "slice_grid": {
        "zh_tw": "網格搜尋切片",
        "zh_cn": "网格搜索切片",
        "en": "Grid Search",
        "ja": "グリッドサーチ",
    },
    "export_format": {
        "zh_tw": "匯出格式",
        "zh_cn": "导出格式",
        "en": "Export",
        "ja": "出力形式",
    },
    "tempo_bpm": {
        "zh_tw": "速度 BPM",
        "zh_cn": "Tempo BPM",
        "en": "Tempo BPM",
        "ja": "テンポ (BPM)",
    },
    "quant_step": {
        "zh_tw": "MIDI 量化精度",
        "zh_cn": "MIDI 量化精度",
        "en": "Quantization",
        "ja": "MIDI クオンタイズ",
    },
    "quant_off": {
        "zh_tw": "不量化",
        "zh_cn": "不量化",
        "en": "Off",
        "ja": "なし",
    },
    "quant_1_4": {
        "zh_tw": "1/4 音符",
        "zh_cn": "1/4 音符",
        "en": "1/4 note",
        "ja": "4分音符",
    },
    "quant_1_8": {
        "zh_tw": "1/8 音符",
        "zh_cn": "1/8 音符",
        "en": "1/8 note",
        "ja": "8分音符",
    },
    "quant_1_16": {
        "zh_tw": "1/16 音符",
        "zh_cn": "1/16 音符",
        "en": "1/16 note",
        "ja": "16分音符",
    },
    "quant_1_32": {
        "zh_tw": "1/32 音符",
        "zh_cn": "1/32 音符",
        "en": "1/32 note",
        "ja": "32分音符",
    },
    "quant_1_64": {
        "zh_tw": "1/64 音符",
        "zh_cn": "1/64 音符",
        "en": "1/64 note",
        "ja": "64分音符",
    },
    "output_lyrics": {
        "zh_tw": "輸出歌詞",
        "zh_cn": "输出歌词",
        "en": "Output Lyrics",
        "ja": "歌詞を出力",
    },
    "save_dir": {
        "zh_tw": "儲存目錄",
        "zh_cn": "保存目录",
        "en": "Save Directory",
        "ja": "保存先ディレクトリ",
    },
    "section_models": {
        "zh_tw": "模型路徑",
        "zh_cn": "模型路径",
        "en": "Model Paths",
        "ja": "モデルパス",
    },
    "game_path": {
        "zh_tw": "GAME 模型路徑",
        "zh_cn": "GAME 模型路径",
        "en": "GAME Model Path",
        "ja": "GAME モデルパス",
    },
    "tifa_path": {
        "zh_tw": "TiFA 模型路徑",
        "zh_cn": "TiFA 模型路径",
        "en": "TiFA Model Path",
        "ja": "TiFA モデルパス",
    },
    "hfa_path": {
        "zh_tw": "HubertFA 模型路徑",
        "zh_cn": "HubertFA 模型路径",
        "en": "HubertFA Model Path",
        "ja": "HubertFA モデルパス",
    },
    "asr_path": {
        "zh_tw": "Qwen3-ASR 模型路徑",
        "zh_cn": "Qwen3-ASR 模型路径",
        "en": "Qwen3-ASR Model Path",
        "ja": "Qwen3-ASR モデルパス",
    },
    "phoneme_path": {
        "zh_tw": "音素 ASR 模型路徑",
        "zh_cn": "音素 ASR 模型路径",
        "en": "Phoneme ASR Model Path",
        "ja": "音素 ASR モデルパス",
    },
    "pinyin_path": {
        "zh_tw": "拼音 ASR 模型路徑",
        "zh_cn": "拼音 ASR 模型路径",
        "en": "Pinyin ASR Model Path",
        "ja": "ピンイン ASR モデルパス",
    },
    "kashi_g2p_path": {
        "zh_tw": "kashi-g2p 模型路徑",
        "zh_cn": "kashi-g2p 模型路径",
        "en": "kashi-g2p Model Path",
        "ja": "kashi-g2p モデルパス",
    },
    "rmvpe_path": {
        "zh_tw": "RMVPE 模型路徑",
        "zh_cn": "RMVPE 模型路径",
        "en": "RMVPE Model Path",
        "ja": "RMVPE モデルパス",
    },
    "ui_language": {
        "zh_tw": "介面語言",
        "zh_cn": "界面语言",
        "en": "UI Language",
        "ja": "表示言語",
    },
    "run": {
        "zh_tw": "開始轉換",
        "zh_cn": "开始提取",
        "en": "Start",
        "ja": "変換開始",
    },
    "stop": {
        "zh_tw": "停止",
        "zh_cn": "停止",
        "en": "Stop",
        "ja": "停止",
    },
    "stopping": {
        "zh_tw": "正在停止...",
        "zh_cn": "正在停止...",
        "en": "Stopping...",
        "ja": "停止中...",
    },
    "idle": {
        "zh_tw": "就緒",
        "zh_cn": "就绪",
        "en": "Ready",
        "ja": "待機中",
    },
    "preparing": {
        "zh_tw": "準備處理 {n} 個檔案...",
        "zh_cn": "准备处理 {n} 个文件...",
        "en": "Preparing {n} file(s)...",
        "ja": "{n} 個のファイルを準備中...",
    },
    "processing": {
        "zh_tw": "處理中 {i}/{n}: {f}",
        "zh_cn": "处理中 {i}/{n}: {f}",
        "en": "Processing {i}/{n}: {f}",
        "ja": "処理中 {i}/{n}: {f}",
    },
    "log_title": {
        "zh_tw": "執行日誌",
        "zh_cn": "运行日志",
        "en": "Log",
        "ja": "ログ",
    },
    "err_no_audio": {
        "zh_tw": "請至少選擇一個音訊檔案",
        "zh_cn": "请至少选择一个音频文件",
        "en": "Please add at least one audio file",
        "ja": "音声ファイルを1つ以上選択してください",
    },
    "err_hybrid": {
        "zh_tw": "錯誤: 推論管線載入失敗: {err}",
        "zh_cn": "错误: 推理管线未能加载: {err}",
        "en": "Error: inference pipeline failed to load: {err}",
        "ja": "エラー: 推論パイプラインの読み込みに失敗しました: {err}",
    },
    "err_busy": {
        "zh_tw": "任務正在執行中",
        "zh_cn": "任务正在运行",
        "en": "A task is already running",
        "ja": "タスクを実行中です",
    },
    "worker_processing": {
        "zh_tw": "====== 正在處理: {f} ======",
        "zh_cn": "====== 正在处理: {f} ======",
        "en": "====== Processing: {f} ======",
        "ja": "====== 処理中: {f} ======",
    },
    "worker_success": {
        "zh_tw": "轉換成功！檔案已儲存至: {d}",
        "zh_cn": "提取成功！文件已保存至: {d}",
        "en": "Done! Files saved to: {d}",
        "ja": "変換完了！保存先: {d}",
    },
    "worker_cancelled": {
        "zh_tw": "任務已被取消。",
        "zh_cn": "任务已被取消。",
        "en": "Task was cancelled.",
        "ja": "タスクがキャンセルされました。",
    },
    "worker_stopped": {
        "zh_tw": "任務已被停止。",
        "zh_cn": "任务已被停止。",
        "en": "Task was stopped.",
        "ja": "タスクが停止されました。",
    },
    "worker_error": {
        "zh_tw": "發生錯誤:\n{tb}",
        "zh_cn": "发生错误:\n{tb}",
        "en": "An error occurred:\n{tb}",
        "ja": "エラーが発生しました:\n{tb}",
    },
}

# (backend value, i18n key) option tables, shared by main.py / the .kv file.
TARGET_LANGUAGE_CHOICES = [
    ("zh", "lang_zh"),
    ("ja", "lang_ja"),
    ("en", "lang_en"),
    ("yue", "lang_yue"),
]

SLICE_METHOD_CHOICES = [
    ("smart", "slice_smart"),
    ("heuristic", "slice_heuristic"),
    ("default", "slice_default"),
    ("grid", "slice_grid"),
]

EXPORT_FORMAT_CHOICES = [
    ("mid", "MIDI"),
    ("ustx", "USTX"),
    ("vsqx", "VSQX"),
]

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
    """Switch active UI language: zh_tw (default), ja, en, or zh_cn."""
    global current_language
    value = str(language or "").strip().lower()
    if value in {"zh_tw", "tw", "hant", "繁體", "繁體中文"}:
        current_language = "zh_tw"
    elif value in {"ja", "jp", "japanese", "日文", "日本語"}:
        current_language = "ja"
    elif value in {"en", "english", "英文", "英語"}:
        current_language = "en"
    elif value in {"zh_cn", "zh", "cn", "hans", "簡體", "简体", "简体中文"}:
        current_language = "zh_cn"
    else:
        current_language = DEFAULT_LANGUAGE
    return current_language


def get_language() -> str:
    return current_language


def tr(key: str, **kwargs) -> str:
    """Return the translated string for ``key`` (or ``key`` itself if unknown)."""
    entry = STRINGS.get(key)
    if entry is None:
        return key
    text = (
        entry.get(current_language)
        or entry.get(DEFAULT_LANGUAGE)
        or entry.get("zh_cn")
        or entry.get("zh")
        or entry.get("en")
        or key
    )
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text
