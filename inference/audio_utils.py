import wave
import numpy as np

def read_wav(file_path):
    with wave.open(str(file_path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        
        if sampwidth == 1:
            dtype = np.uint8
        elif sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sampwidth {sampwidth}")
            
        audio = np.frombuffer(data, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            
        return framerate, audio

def write_wav(file_path, rate, data):
    with wave.open(str(file_path), 'wb') as wf:
        if data.ndim == 1:
            n_channels = 1
        else:
            n_channels = data.shape[1]
            
        # Standardize to int16 for output
        if data.dtype == np.float32 or data.dtype == np.float64:
            # Assumes float data is in [-1.0, 1.0]
            data = np.clip(data, -1.0, 1.0)
            data = (data * 32767).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)
            
        wf.setnchannels(n_channels)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())

def resample_numpy(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
        
    duration = len(audio) / orig_sr
    n_samples_target = int(np.round(duration * target_sr))
    
    x = np.arange(len(audio)) / orig_sr
    x_new = np.arange(n_samples_target) / target_sr
    
    if audio.ndim == 1:
        resampled = np.interp(x_new, x, audio)
    else:
        resampled = np.zeros((n_samples_target, audio.shape[1]), dtype=audio.dtype)
        for i in range(audio.shape[1]):
            resampled[:, i] = np.interp(x_new, x, audio[:, i])
            
    return resampled.astype(np.float32)

def get_window_numpy(window_type, Nx, fftbins=True):
    if window_type == "hann":
        if fftbins:
            return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(Nx) / Nx)
        else:
            return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(Nx) / (Nx - 1))
    raise ValueError(f"Unsupported window type {window_type}")
