"""MDX-Net vocal separation (UVR ONNX models) in pure numpy + onnxruntime.

usage: python separate_vocals.py in.wav(44.1k stereo) out_vocals.wav model.onnx [compensate]
"""
import sys
import numpy as np
import soundfile as sf
import onnxruntime as ort

N_FFT, HOP, DIM_F, DIM_T = 7680, 1024, 3072, 256
N_BINS = N_FFT // 2 + 1
CHUNK = HOP * (DIM_T - 1)
WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float64)  # periodic hann


def stft(x):  # x: [n, CHUNK] -> [n, N_BINS, DIM_T] complex (torch.stft center=True, reflect)
    pad = N_FFT // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    idx = np.arange(DIM_T)[:, None] * HOP + np.arange(N_FFT)[None, :]
    frames = xp[:, idx] * WINDOW  # [n, DIM_T, N_FFT]
    spec = np.fft.rfft(frames, axis=-1)  # [n, DIM_T, N_BINS]
    return spec.transpose(0, 2, 1)


def istft(spec):  # [n, N_BINS, DIM_T] -> [n, CHUNK]
    frames = np.fft.irfft(spec.transpose(0, 2, 1), n=N_FFT, axis=-1) * WINDOW
    total = N_FFT + HOP * (DIM_T - 1)
    out = np.zeros((spec.shape[0], total))
    wsum = np.zeros(total)
    for t in range(DIM_T):
        out[:, t * HOP:t * HOP + N_FFT] += frames[:, t]
        wsum[t * HOP:t * HOP + N_FFT] += WINDOW ** 2
    pad = N_FFT // 2
    out = out[:, pad:pad + CHUNK]
    return out / np.maximum(wsum[pad:pad + CHUNK], 1e-8)


def to_model(waves):  # [b, 2, CHUNK] -> [b, 4, DIM_F, DIM_T]
    b = waves.shape[0]
    s = stft(waves.reshape(-1, CHUNK))  # [b*2, bins, T]
    s = np.stack([s.real, s.imag], axis=1)  # [b*2, 2, bins, T]
    s = s.reshape(b, 4, N_BINS, DIM_T)
    return s[:, :, :DIM_F].astype(np.float32)


def from_model(spec):  # [b, 4, DIM_F, DIM_T] -> [b, 2, CHUNK]
    b = spec.shape[0]
    full = np.zeros((b, 4, N_BINS, DIM_T), dtype=np.float64)
    full[:, :, :DIM_F] = spec
    full = full.reshape(b * 2, 2, N_BINS, DIM_T)
    c = full[:, 0] + 1j * full[:, 1]
    return istft(c).reshape(b, 2, CHUNK)


def main():
    src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
    compensate = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    mix, sr = sf.read(src, always_2d=True)
    assert sr == 44100, sr
    mix = mix.T.astype(np.float64)  # [2, N]
    n = mix.shape[1]
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    trim = N_FFT // 2
    gen = CHUNK - 2 * trim
    pad = gen - n % gen
    mp = np.concatenate([np.zeros((2, trim)), mix, np.zeros((2, pad)), np.zeros((2, trim))], 1)
    waves = np.stack([mp[:, i:i + CHUNK] for i in range(0, n + pad, gen)])
    outs = []
    for k, w in enumerate(waves):
        spek = to_model(w[None])
        # "denoise" trick from UVR: average f(x) and -f(-x) to cancel model noise
        a = sess.run(None, {"input": spek})[0]
        b = sess.run(None, {"input": -spek})[0]
        outs.append(from_model((a - b) * 0.5)[0])
        print(f"chunk {k + 1}/{len(waves)}", flush=True)
    outs = np.stack(outs)[:, :, trim:-trim]  # [k, 2, gen]
    voc = outs.transpose(1, 0, 2).reshape(2, -1)[:, :n] * compensate
    sf.write(dst, voc.T, sr, subtype="FLOAT")
    sf.write(dst.replace(".wav", "_inst.wav"), (mix - voc).T, sr, subtype="FLOAT")


if __name__ == "__main__":
    main()
