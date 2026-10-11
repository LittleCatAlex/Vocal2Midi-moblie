[app]

title = Vocal2Midi
package.name = vocal2midi
package.domain = org.vocal2midi

source.dir = .
source.include_exts = py,kv,png,jpg,atlas,onnx,txt,json,yaml,ttf,otf
# Desktop-only code / assets / huge or Windows-only files stay out of the APK.
source.exclude_dirs = python, gui, scripts, tests, models, outputs, settings, .git, .venv, __pycache__, bin, inference/qwen3asr_dml
source.exclude_patterns = *.dll, *.bat, *.exe, *.pyc, install.bat, run.bat, app_fluent.py

version = 0.1.0

# NOTE (see Build_Report.md): "" has NO python-for-android recipe and
# there are no Android wheels on PyPI, so it must come from ./p4a-recipes (not
# provided yet). librosa / soundfile / numba / tokenizers / pyopenjtalk are
# intentionally NOT listed: they have no p4a recipes (llvmlite, libsndfile,
# Rust, C++ toolchains) and the code must first be ported to numpy/scipy/av.
# pure-Python packages (ruamel.yaml, tqdm, mido, textgrid, opencc) install as-is.
requirements = hostpython3==3.11.9,python3==3.11.9,kivy,plyer,pyjnius,numpy,ruamel.yaml,tqdm,mido,textgrid,opencc-python-reimplemented

# Custom recipes (e.g. an  recipe) go here.
p4a.local_recipes = ./p4a-recipes

orientation = portrait
fullscreen = 0

android.permissions = READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE, READ_MEDIA_AUDIO, MANAGE_EXTERNAL_STORAGE, INTERNET

# API 33 so READ_MEDIA_AUDIO exists; main.py requests it at runtime (API>=33) and
# asks for "All files access" (API>=30) so /storage/emulated/0/Vocal2Midi works.
android.api = 33
android.minapi = 24
android.ndk = 28c
android.accept_sdk_license = True
android.archs = arm64-v8a
android.allow_backup = True
android.enable_androidx = True
android.add_aars = libs/onnxruntime-android-1.17.1.aar
android.add_gradle_repositories = flatDir { dirs 'libs' }
p4a.python_version = 3.11

[buildozer]
log_level = 2
warn_on_root = 1
