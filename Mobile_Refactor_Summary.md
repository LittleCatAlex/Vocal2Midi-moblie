# Mobile_Refactor_Summary

Android-oriented refactor of Vocal2Midi (PySide6 + qfluentwidgets -> Kivy). The original `gui/`, `app_fluent.py` and `run.bat` are **left untouched** so the desktop build still works; the new mobile entry is `main.py`.

## New / changed files

| File | Status | Purpose |
|---|---|---|
| `main.py` | new | Kivy `App`, plyer file picker, `threading` worker, `Clock.schedule_once` UI updates |
| `vocal2midi.kv` | new | Single-screen layout (replaces all Qt views/dialogs) |
| `i18n.py` (project root) | new | Pure-Python zh/en dict lookup (replaces `gui/i18n.py`) |
| `inference/device_utils.py` | edited | All ONNX provider resolution forced to `CPUExecutionProvider` |
| `inference/kashi_g2p_ja/g2p_onnx_runtime.py` | edited | Direct `InferenceSession` no longer tries DirectML |
| `tests/headless_smoke_test.py` | new | No-UI smoke test |

## Qt component mapping

| Old (desktop) | New (mobile) |
|---|---|
| `fluent_main.py` (`FluentWindow`, nav sidebar, close-confirm dialog) | Root `BoxLayout` + `ScrollView` in the `.kv`; no navigation, no close dialog |
| `auto_lyric_view.py` (main page) | Top part of the `.kv` + `Vocal2MidiApp` methods in `main.py` (`_collect_values`, `build_config`, `start_pipeline`) |
| `audio_file_list.py` (table, drag & drop) | `audio_summary` label + `plyer.filechooser.open_file(multiple=True)` |
| `model_config_view.py` | "Model Paths" `TextInput`s (defaults under `<storage>/Vocal2Midi/models/`) |
| `global_settings_view.py` | **Dropped** (theme, batch mode, advanced params); only UI-language toggle kept; advanced params use `PipelineConfig` defaults (`t0=0`, `nsteps=8`) |
| `file_settings_dialog.py`, `bpm_spinbox.py` | **Dropped** (per-file settings, BPM drag&drop detect); plain `tempo_input` |
| `log_terminal.py` | `log_label` in the `.kv` (last 300 lines) |
| `option_tables.py`, `fluent_utils.py`, `settings_utils.py` | Choice tables moved into `i18n.py`, `t0_nstep_to_ts` into `main.py`; `QSettings` persistence dropped |
| `fluent_worker.py` (`QThread`, `Signal`) | `threading.Thread` + `threading.Event` (stop) + `Clock.schedule_once` |
| `app_fluent.py` | `main.py` (`python main.py` / Buildozer entry) |

## Threading model
- `start_pipeline` (UI thread) snapshots widget values into a plain dict, then starts a daemon `threading.Thread`.
- The worker never touches widgets: `log()`, `_set_status()` and `_set_running()` all go through `Clock.schedule_once`. Progress bar = `(finished files / total) * 100`.
- Stop = `threading.Event`, exposed to the pipeline as `cancel_checker` (same mechanism as before; stopping waits for the current model call to return).
- `logging` records and `print` output are forwarded to the on-screen log, as in the old worker.
- plyer's callback may arrive on a non-UI thread, so selection results are marshalled with `Clock.schedule_once` too.

## Windows decoupling
- `run.bat` env vars (`PYTHONHOME`, `PYTHONPATH`, `PATH`, `QT_*`, `V2M_PORTABLE_ROOT`) are **not used** by `main.py`; storage paths come from Android shared storage / `App.user_data_dir`.
- `resolve_onnx_providers()` always returns `("cpu", ["CPUExecutionProvider"])`; `normalize_runtime_device()` always returns `"cpu"`; `use_dml()` is `False`. This covers GAME, HubertFA, TiFA, RMVPE, Romaji/Pinyin ASR and the Qwen3 encoder, which all go through that function. `kashi-g2p` and the Qwen3 decoders were already/now explicitly CPU.
- `PipelineConfig.device` is always `"cpu"`.

## Known gaps before a real Buildozer build (not done here)
1. `inference/qwen3asr_dml/` ships Windows `.dll` llama.cpp binaries (`llama.py`, `bin/`); the Qwen3-ASR (llama.cpp) path will **not** work on Android without ARM `.so` builds. Romaji/Pinyin ASR (pure ONNX) are the realistic options on-device.
2. `onnxruntime`, `numpy`, `librosa`/`soundfile` etc. need Android recipes (python-for-android has none for onnxruntime out of the box); `buildozer.spec` still has to be written (requirements: `python3,kivy,plyer,numpy,...`).
3. Qwen/model sizes are multi-GB; models must be copied to shared storage manually (plyer's `filechooser` cannot pick directories on Android, hence text path fields).
4. The legacy `pyopenjtalk` / DirectML-oriented comments and `device_utils` DXGI code remain (dead code on Android, harmless).
5. Tested only on desktop (Kivy 2.3.1, hidden window, stubbed pipeline) via `python tests/headless_smoke_test.py` - all checks pass; not tested on a device or with real models.
