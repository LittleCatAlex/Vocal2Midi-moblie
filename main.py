"""Vocal2Midi - Kivy mobile entry point (Android / Buildozer).

Replaces the former Qt desktop GUI:
  * layout lives in ``vocal2midi.kv``;
  * files are picked through the native system picker (``plyer``);
  * heavy work runs on a plain ``threading.Thread`` and talks to the UI only
    through ``Clock.schedule_once``;
  * ONNX inference is CPU-only (see ``inference/device_utils.py``).
"""
from __future__ import annotations

import logging
import os
import pathlib
import sys
import threading
import traceback

from kivy.app import App
from kivy.clock import Clock

from inference.ort_android_wrapper import inject_ort_mock
inject_ort_mock()

from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.utils import platform

import i18n
from i18n import tr

ROOT_DIR = pathlib.Path(__file__).resolve().parent
KV_FILE = ROOT_DIR / "vocal2midi.kv"

CJK_FONT_CANDIDATES = [
    # Android system fonts
    "/system/fonts/NotoSansCJK-Regular.ttc",
    "/system/fonts/NotoSansSC-Regular.otf",
    "/system/fonts/NotoSansCJKsc-Regular.otf",
    "/system/fonts/DroidSansFallback.ttf",
    # Desktop (dev/testing)
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def register_cjk_font() -> str | None:
    """Replace Kivy's default Roboto (no CJK glyphs) with a system CJK font."""
    from kivy.core.text import LabelBase

    bundled = ROOT_DIR / "fonts" / "CJK.ttf"  # optional: ship your own font here
    for candidate in [str(bundled)] + CJK_FONT_CANDIDATES:
        if os.path.isfile(candidate):
            LabelBase.register(name="Roboto", fn_regular=candidate, fn_bold=candidate)
            return candidate
    return None


AUDIO_PATTERNS = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aac"]

DEFAULT_MODEL_SUBDIRS = {
    "game_model": "GAME-1.0.3-medium-onnx",
    "tifa_model": "tifa-1.0-onnx",
    "hfa_model": "1218_hfa_model_new_dict",
    "asr_model": "Qwen3-ASR-1.7B-dml",
    "phoneme_asr_model": "romajiASR",
    "pinyin_asr_model": "pinyinASR",
    "kashi_g2p_model": "kashi-g2p-onnx",
    "rmvpe_model": "RMVPE",
}


def t0_nstep_to_ts(t0: float, nsteps: int) -> list:
    """Sampling schedule for the D3PM decoder (was gui/fluent_utils.py)."""
    if nsteps <= 0:
        return [t0]
    step = (1 - t0) / nsteps
    return [t0 + i * step for i in range(nsteps)]


ANDROID_PUBLIC_ROOT = pathlib.Path("/storage/emulated/0/Vocal2Midi")


def storage_root(app: App | None = None) -> pathlib.Path:
    """Root for models / outputs: public shared storage on Android.

    Models go in ``<root>/models/<name>/`` and results in ``<root>/outputs/``.
    """
    if platform == "android":
        try:
            from android.storage import primary_external_storage_path  # type: ignore

            return pathlib.Path(primary_external_storage_path()) / "Vocal2Midi"
        except Exception:
            return ANDROID_PUBLIC_ROOT
    if app is not None:
        return pathlib.Path(app.user_data_dir)
    return ROOT_DIR / "data"


def model_subdirs() -> dict:
    """Model folders for this platform (Qwen3-ASR is not used on Android)."""
    if platform == "android":
        return {k: v for k, v in DEFAULT_MODEL_SUBDIRS.items() if k != "asr_model"}
    return dict(DEFAULT_MODEL_SUBDIRS)


class _UiLogHandler(logging.Handler):
    """Forward logging records from the worker thread to the UI thread."""

    def __init__(self, sink):
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record):
        try:
            self.sink(record.getMessage())
        except Exception:
            self.handleError(record)


class _StreamRedirector:
    """Mirror print() output into the on-screen log."""

    def __init__(self, stream, sink):
        self.stream = stream
        self.sink = sink

    def write(self, text):
        if text and text.strip():
            self.sink(text.strip())
        if self.stream is not None:
            self.stream.write(text)

    def flush(self):
        if self.stream is not None:
            self.stream.flush()


class Vocal2MidiApp(App):
    lang = StringProperty(i18n.get_language())
    progress = NumericProperty(0)
    status_text = StringProperty("")
    is_running = BooleanProperty(False)
    audio_summary = StringProperty("")

    title = "Vocal2Midi"

    # ------------------------------------------------------------------ setup
    def build(self):
        register_cjk_font()
        # Android has no Qwen3-ASR, so only zh (PinyinASR) / ja (RomajiASR) are offered.
        self.language_choices = [
            c for c in i18n.TARGET_LANGUAGE_CHOICES if platform != "android" or c[0] in ("zh", "ja")
        ]
        self.audio_files: list[str] = []
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._log_lines: list[str] = []
        self.status_text = tr("idle")
        self._update_audio_summary()
        root = Builder.load_file(str(KV_FILE))
        self.root = root  # App.run() assigns this later; helpers need it now
        self._init_defaults(root)
        return root

    def on_start(self):
        if platform == "android":
            self._request_android_permissions()
        else:
            self.ensure_storage_dirs()

    def _request_android_permissions(self):
        """Ask for storage / audio permissions, then create the public dirs."""
        try:
            from android.permissions import request_permissions  # type: ignore
            from android import api_version  # type: ignore

            if api_version >= 33:
                # Android 13+: READ/WRITE_EXTERNAL_STORAGE are no-ops; audio is granular.
                perms = ["android.permission.READ_MEDIA_AUDIO"]
            else:
                perms = [
                    "android.permission.READ_EXTERNAL_STORAGE",
                    "android.permission.WRITE_EXTERNAL_STORAGE",
                ]
            request_permissions(perms, self._on_permissions_result)
        except Exception as exc:  # pragma: no cover - android only
            self.log(f"permissions: {exc}")
        self._request_all_files_access()
        self.ensure_storage_dirs()

    def _on_permissions_result(self, permissions, grants):
        # May run off the UI thread; directory creation touches no widgets.
        self.ensure_storage_dirs()
        if not all(grants):
            self.log("Storage permission was not granted; model/output folders may be unavailable.")

    def _request_all_files_access(self):
        """Android 11+: /storage/emulated/0/Vocal2Midi needs 'All files access'."""
        try:
            from jnius import autoclass  # type: ignore

            Build = autoclass("android.os.Build$VERSION")
            if Build.SDK_INT < 30:
                return
            Environment = autoclass("android.os.Environment")
            if Environment.isExternalStorageManager():
                return
            Intent = autoclass("android.content.Intent")
            Settings = autoclass("android.provider.Settings")
            Uri = autoclass("android.net.Uri")
            activity = autoclass("org.kivy.android.PythonActivity").mActivity
            intent = Intent(
                Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                Uri.parse("package:" + activity.getPackageName()),
            )
            activity.startActivity(intent)
        except Exception as exc:  # pragma: no cover - android only
            self.log(f"all-files-access: {exc}")

    def ensure_storage_dirs(self):
        """Create <root>/models/<each model dir> and <root>/outputs if missing."""
        base = storage_root(self)
        created = []
        try:
            for sub in [pathlib.Path("outputs")] + [
                pathlib.Path("models") / name for name in model_subdirs().values()
            ]:
                target = base / sub
                if not target.exists():
                    target.mkdir(parents=True, exist_ok=True)
                    created.append(str(target))
        except Exception as exc:
            self.log(f"cannot create {base}: {exc}")
        return created

    def _init_defaults(self, root):
        base = storage_root(self)
        ids = root.ids
        ids.save_dir.text = str(base / "outputs")
        for key, sub in DEFAULT_MODEL_SUBDIRS.items():
            widget = ids.get(f"path_{key}")
            if widget is not None:
                widget.text = str(base / "models" / sub)
        self.refresh_choices()

    # -------------------------------------------------------------- i18n/UI
    def tr(self, lang, key, **kwargs):
        """kv helper: ``app.tr(app.lang, 'key')`` re-evaluates on language change."""
        return i18n.tr(key, **kwargs)

    def _label(self, key: str) -> str:
        # Entries without a translation (e.g. "MIDI", "zh") display as-is.
        return tr(key)

    def refresh_choices(self):
        """(Re)fill spinners, keeping the selected backend value."""
        ids = self.root.ids
        for spinner_id, choices in (
            ("slice_spinner", i18n.SLICE_METHOD_CHOICES),
            ("quant_spinner", i18n.QUANT_STEP_CHOICES),
            ("format_spinner", i18n.EXPORT_FORMAT_CHOICES),
            ("lang_spinner", self.language_choices),
        ):
            spinner = ids[spinner_id]
            previous = getattr(spinner, "_backend_value", None)
            spinner.values = [self._label(label) for _, label in choices]
            index = 0
            if previous is not None:
                for i, (value, _) in enumerate(choices):
                    if value == previous:
                        index = i
                        break
            spinner.text = spinner.values[index]
            spinner._backend_value = choices[index][0]

    def on_choice_selected(self, spinner, text, choices):
        for value, label in choices:
            if self._label(label) == text:
                spinner._backend_value = value
                return

    def set_ui_language(self, code: str):
        self.lang = i18n.set_language(code)
        self.refresh_choices()
        if not self.is_running:
            self.status_text = tr("idle")
        self._update_audio_summary()

    # ------------------------------------------------------- file selection
    def open_file_picker(self):
        """Native file picker via plyer (no drag & drop on mobile)."""
        try:
            from plyer import filechooser

            filters = ["audio/*"] if platform == "android" else AUDIO_PATTERNS
            filechooser.open_file(
                on_selection=self._on_files_selected,
                multiple=True,
                filters=filters,
                title=tr("pick_files"),
            )
        except Exception as exc:
            self.log(tr("picker_unavailable", err=exc))

    def _on_files_selected(self, selection):
        # plyer may invoke this from a non-UI thread (Android activity result).
        paths = list(selection or [])
        Clock.schedule_once(lambda _dt: self._add_files(paths), 0)

    def _add_files(self, paths):
        added = 0
        for path in paths:
            path = str(path)
            if path not in self.audio_files:
                self.audio_files.append(path)
                added += 1
        self._update_audio_summary()
        if added:
            self.log(tr("files_added", n=added))

    def clear_files(self):
        self.audio_files.clear()
        self._update_audio_summary()

    def _update_audio_summary(self):
        if not self.audio_files:
            self.audio_summary = tr("no_files")
        else:
            self.audio_summary = "\n".join(
                f"{i}. {pathlib.Path(p).name}" for i, p in enumerate(self.audio_files, 1)
            )

    # ----------------------------------------------------------------- log
    def log(self, message: str):
        """Thread-safe: may be called from any thread."""
        Clock.schedule_once(lambda _dt: self._append_log(str(message)), 0)

    def _append_log(self, message: str):
        self._log_lines.append(message)
        self._log_lines = self._log_lines[-300:]
        if self.root is not None:
            self.root.ids.log_label.text = "\n".join(self._log_lines)

    def _set_status(self, text: str, progress: float | None = None):
        def _apply(_dt):
            self.status_text = text
            if progress is not None:
                self.progress = progress

        Clock.schedule_once(_apply, 0)

    def _set_running(self, running: bool):
        Clock.schedule_once(lambda _dt: setattr(self, "is_running", running), 0)

    # ------------------------------------------------------- config build
    def _collect_values(self) -> dict:
        ids = self.root.ids

        def _float(widget, default):
            try:
                return float(widget.text)
            except (TypeError, ValueError):
                return default

        language = ids.lang_spinner._backend_value
        return {
            "language": language,
            "slicing_method": ids.slice_spinner._backend_value,
            "quantization_step": ids.quant_spinner._backend_value,
            "export_format": ids.format_spinner._backend_value,
            "tempo": _float(ids.tempo_input, 120.0),
            "output_lyrics": bool(ids.output_lyrics_check.active),
            "original_lyrics": ids.lyrics_input.text.strip(),
            "output_dir": ids.save_dir.text.strip(),
            "models": {
                key: ids[f"path_{key}"].text.strip() for key in DEFAULT_MODEL_SUBDIRS
            },
        }

    def build_pipeline_config(self, values: dict):
        from application.config import PipelineConfig

        models = values["models"]
        language = values["language"]
        export_format = values["export_format"]
        return PipelineConfig(
            audio_path="",
            output_filename="",
            output_dir=pathlib.Path(values["output_dir"]),
            game_model_dir=models["game_model"],
            hfa_model_dir=models["hfa_model"],
            asr_model_path=models["asr_model"],
            device="cpu",  # Android: CPUExecutionProvider only
            language=language,
            ts=t0_nstep_to_ts(0.0, 8),
            lyric_output_mode=i18n.DEFAULT_LYRIC_OUTPUT.get(language, "hanzi"),
            original_lyrics=values["original_lyrics"],
            output_formats=[export_format],
            output_lyrics=values["output_lyrics"],
            slicing_method=values["slicing_method"],
            tempo=values["tempo"],
            quantization_step=int(values["quantization_step"]),
            rmvpe_model_path=models["rmvpe_model"],
            phoneme_asr_model_path=models["phoneme_asr_model"],
            pinyin_asr_model_path=models["pinyin_asr_model"],
            kashi_g2p_model_path=models["kashi_g2p_model"],
            tifa_model_path=models["tifa_model"],
        )

    # -------------------------------------------------------- run / stop
    def start_pipeline(self):
        if self.is_running:
            self.log(tr("err_busy"))
            return
        if not self.audio_files:
            self.log(tr("err_no_audio"))
            return
        values = self._collect_values()
        files = list(self.audio_files)
        self._stop_event.clear()
        self.progress = 0
        self.is_running = True
        self.status_text = tr("preparing", n=len(files))
        self._worker = threading.Thread(
            target=self._run_worker, args=(values, files), name="v2m-worker", daemon=True
        )
        self._worker.start()

    def stop_pipeline(self):
        if self.is_running:
            self._stop_event.set()
            self.status_text = tr("stopping")

    def _run_worker(self, values: dict, files: list[str]):
        """Runs on a background thread. Never touch widgets directly here."""
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _StreamRedirector(old_stdout, self.log)
        sys.stderr = _StreamRedirector(old_stderr, self.log)
        root_logger = logging.getLogger()
        old_level = root_logger.level
        handler = _UiLogHandler(self.log)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        session = None
        try:
            try:
                from application.exceptions import CancellationError
                from application.pipeline import open_asr_session, run_auto_lyric_job
            except Exception as exc:
                self.log(tr("err_hybrid", err=exc))
                return

            os.makedirs(values["output_dir"], exist_ok=True)
            session = open_asr_session()
            total = len(files)
            try:
                for index, path in enumerate(files, start=1):
                    if self._stop_event.is_set():
                        break
                    name = pathlib.Path(path).name
                    self.log(tr("worker_processing", f=name))
                    self._set_status(
                        tr("processing", i=index, n=total, f=name),
                        (index - 1) / total * 100,
                    )
                    config = self.build_pipeline_config(values)
                    config.audio_path = path
                    config.output_filename = name
                    config.asr_session = session
                    config.cancel_checker = self._stop_event.is_set
                    run_auto_lyric_job(config)
                    self._set_status(tr("processing", i=index, n=total, f=name), index / total * 100)
                if self._stop_event.is_set():
                    self.log(tr("worker_cancelled"))
                    self._set_status(tr("worker_cancelled"))
                else:
                    self.log(tr("worker_success", d=values["output_dir"]))
                    self._set_status(tr("worker_success", d=values["output_dir"]), 100)
            except (InterruptedError, CancellationError):
                self.log(tr("worker_stopped"))
                self._set_status(tr("worker_stopped"))
        except Exception:
            tb = traceback.format_exc()
            self.log(tr("worker_error", tb=tb))
            self._set_status(tr("worker_error", tb="").strip())
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass
            sys.stdout, sys.stderr = old_stdout, old_stderr
            self._set_running(False)


def main():
    Vocal2MidiApp().run()


if __name__ == "__main__":
    main()
