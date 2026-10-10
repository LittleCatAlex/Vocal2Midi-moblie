import os

files_to_update = {
    'inference/API/asr_api.py': [
        (b'import scipy.io.wavfile as wavfile', b'from inference.audio_utils import write_wav'),
        (b'wavfile.write(chunk_path, sr, chunk["waveform"])', b'write_wav(chunk_path, sr, chunk["waveform"])')
    ],
    'inference/pipeline/auto_lyric_hybrid.py': [
        (b'import scipy.io.wavfile as wavfile', b'from inference.audio_utils import write_wav'),
        (b'wavfile.write(output_dir / f"{output_key}_{chunk_idx:03d}.wav", sr, chunk["waveform"])', b'write_wav(output_dir / f"{output_key}_{chunk_idx:03d}.wav", sr, chunk["waveform"])')
    ],
    'inference/romaji_asr/common.py': [
        (b'import scipy.io.wavfile as wavfile\r\nfrom scipy.signal import resample_poly', b'from inference.audio_utils import read_wav, resample_numpy'),
        (b'import scipy.io.wavfile as wavfile\nfrom scipy.signal import resample_poly', b'from inference.audio_utils import read_wav, resample_numpy'),
        (b'import scipy.io.wavfile as wavfile', b'from inference.audio_utils import read_wav'),
        (b'from scipy.signal import resample_poly', b'from inference.audio_utils import resample_numpy'),
        (b'sr, audio = wavfile.read(str(audio_path))', b'sr, audio = read_wav(str(audio_path))'),
        (b'audio = resample_poly(audio, sample_rate, sr).astype(np.float32, copy=False)', b'audio = resample_numpy(audio, sr, sample_rate)')
    ],
    'inference/TiFA/aligner.py': [
        (b'from scipy.signal import resample_poly\r\nimport scipy.io.wavfile as wavfile', b'from inference.audio_utils import read_wav, resample_numpy'),
        (b'from scipy.signal import resample_poly\nimport scipy.io.wavfile as wavfile', b'from inference.audio_utils import read_wav, resample_numpy'),
        (b'from scipy.signal import resample_poly', b'from inference.audio_utils import resample_numpy'),
        (b'import scipy.io.wavfile as wavfile', b'from inference.audio_utils import read_wav'),
        (b'sr, waveform = wavfile.read(str(path))', b'sr, waveform = read_wav(str(path))'),
        (b'waveform = resample_poly(waveform, target_sr, sr).astype(np.float32, copy=False)', b'waveform = resample_numpy(waveform, sr, target_sr)')
    ],
    'inference/API/rmvpe_api.py': [
        (b'from scipy.signal import resample_poly\r\nfrom scipy.signal import get_window', b'from inference.audio_utils import resample_numpy, get_window_numpy'),
        (b'from scipy.signal import resample_poly\nfrom scipy.signal import get_window', b'from inference.audio_utils import resample_numpy, get_window_numpy'),
        (b'from scipy.signal import resample_poly', b'from inference.audio_utils import resample_numpy, get_window_numpy'),
        (b'from scipy.signal import get_window\r\n', b''),
        (b'from scipy.signal import get_window\n', b''),
        (b'from scipy.signal import get_window', b''),
        (b'resample_poly(waveform, SAMPLE_RATE, sample_rate)', b'resample_numpy(waveform, sample_rate, SAMPLE_RATE)'),
        (b'get_window("hann"', b'get_window_numpy("hann"')
    ],
    'inference/qwen3asr_dml/encoder.py': [
        (b'import scipy.signal', b'from inference.audio_utils import get_window_numpy'),
        (b'scipy.signal.get_window("hann", self.n_fft, fftbins=True)', b'get_window_numpy("hann", self.n_fft, fftbins=True)')
    ],
    'inference/qwen3asr_dml/utils.py': [
        (b'import scipy.signal', b'from inference.audio_utils import resample_numpy'),
        (b'scipy.signal.resample_poly(audio, sample_rate, src_rate)', b'resample_numpy(audio, src_rate, sample_rate)')
    ]
}

for filepath, replacements in files_to_update.items():
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            content = f.read()
        for old, new in replacements:
            content = content.replace(old, new)
        with open(filepath, 'wb') as f:
            f.write(content)
        print(f"Updated {filepath}")
    else:
        print(f"File not found: {filepath}")

# Update buildozer.spec
spec_path = 'buildozer.spec'
if os.path.exists(spec_path):
    with open(spec_path, 'rb') as f:
        spec = f.read()
    spec = spec.replace(b'numpy,scipy,', b'numpy,')
    spec = spec.replace(b',scipy', b'')
    with open(spec_path, 'wb') as f:
        f.write(spec)
    print("Updated buildozer.spec")
