"""Headless smoke test: syntax/import checks + kv parse + worker run with a stub pipeline.

Run:  python tests/headless_smoke_test.py
"""
import ast
import os
import pathlib
import sys
import time
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("KIVY_NO_ARGS", "1")
os.environ.setdefault("KIVY_NO_CONSOLELOG", "1")
# Real GL backend + hidden window: the mock backend cannot create an SDL window.
from kivy.config import Config
Config.set("graphics", "window_state", "hidden")

# 1. Syntax check of every touched file --------------------------------------
for rel in [
    "main.py", "i18n.py", "inference/device_utils.py",
    "inference/kashi_g2p_ja/g2p_onnx_runtime.py",
]:
    ast.parse((ROOT / rel).read_text(encoding="utf8"), filename=rel)
print("[ok] syntax")

# 2. No Qt / Windows leftovers in the new entry files ------------------------
for rel in ["main.py", "i18n.py", "vocal2midi.kv"]:
    text = (ROOT / rel).read_text(encoding="utf8")
    for banned in ("PySide6", "qfluentwidgets", "QThread", "Signal(", "DmlExecutionProvider"):
        assert banned not in text, f"{banned} found in {rel}"
print("[ok] no Qt/DirectML references in new files")

# 3. Provider resolution is CPU only (device_utils imports onnxruntime; stub if absent)
try:
    import onnxruntime  # noqa: F401
except ImportError:
    sys.modules["onnxruntime"] = types.SimpleNamespace(get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
from inference.device_utils import normalize_runtime_device, resolve_onnx_providers, use_dml
for dev in (None, "", "dml", "cuda", "gpu", "metal", "cpu"):
    assert resolve_onnx_providers(dev) == ("cpu", ["CPUExecutionProvider"]), dev
    assert normalize_runtime_device(dev) == "cpu"
assert use_dml("dml") is False
print("[ok] ONNX providers forced to CPUExecutionProvider")

# 4. i18n --------------------------------------------------------------------
import i18n
i18n.set_language("en"); assert i18n.tr("run") == "Start"
i18n.set_language("zh"); assert i18n.tr("processing", i=1, n=2, f="a.wav").startswith("处理中 1/2")
assert i18n.tr("missing_key") == "missing_key"
print("[ok] i18n")

# 4b. Android ASR policy (simulated via env var) ------------------------------
from inference import platform_utils as pu
cfg = types.SimpleNamespace(language="zh", chinese_asr_engine="qwen", japanese_asr_engine="qwen", lyric_output_mode="hanzi")
assert pu.apply_android_asr_policy(cfg) == [] or not pu.is_android()  # desktop: untouched
os.environ["ANDROID_ARGUMENT"] = "1"
try:
    assert pu.is_android() and not pu.qwen_asr_available()
    pu.apply_android_asr_policy(cfg)
    assert (cfg.chinese_asr_engine, cfg.lyric_output_mode) == ("pinyin", "pinyin")
    cfg = types.SimpleNamespace(language="ja", chinese_asr_engine="qwen", japanese_asr_engine="qwen", lyric_output_mode="hanzi")
    pu.apply_android_asr_policy(cfg)
    assert (cfg.japanese_asr_engine, cfg.lyric_output_mode) == ("romaji", "romaji")
    cfg.language = "en"
    try:
        pu.apply_android_asr_policy(cfg)
        raise AssertionError("en must be rejected on Android")
    except ValueError:
        pass
finally:
    del os.environ["ANDROID_ARGUMENT"]
print("[ok] Android ASR downgrade policy")

# 5. Kivy: import main, parse kv, build widgets, run worker on stub pipeline ---
from kivy.clock import Clock
import main as v2m

app = v2m.Vocal2MidiApp()
root = app.build()
assert {"save_dir", "log_label", "lang_spinner", "slice_spinner"} <= set(root.ids)
app.root = root
app.set_ui_language("en")
assert root.ids.slice_spinner.text == "Smart"
app.set_ui_language("zh")
print("[ok] kv parsed, widgets built, language switch")

# Stub the heavy pipeline modules so no models/onnx are needed.
calls = []
pipeline = types.ModuleType("application.pipeline")
pipeline.open_asr_session = lambda: types.SimpleNamespace(close=lambda: calls.append("closed"))
def fake_job(cfg):
    assert cfg.device == "cpu"
    calls.append(cfg.output_filename)
pipeline.run_auto_lyric_job = fake_job
sys.modules["application.pipeline"] = pipeline

app._add_files(["/tmp/a.wav", "/tmp/b.wav"])
app.root.ids.save_dir.text = str(ROOT / "tests" / "_out")
app.start_pipeline()
deadline = time.time() + 10
while time.time() < deadline and app.is_running:
    Clock.tick()
    time.sleep(0.02)
for _ in range(5):
    Clock.tick()
assert not app.is_running, "worker did not finish"
assert calls == ["a.wav", "b.wav", "closed"], calls
assert app.progress == 100, app.progress
print("[ok] threaded worker + Clock.schedule_once UI updates:", app.status_text)
print("ALL HEADLESS CHECKS PASSED")
