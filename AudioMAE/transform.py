import torch
import torchaudio.transforms as T


def MelSpectrogram_transform(
                    sample_rate = 16000, 
                    hop_length = 160,
                    n_fft = 400,
                    n_mels = 128,
                    win_length = None,
                    window_fn = torch.hann_window,
                    power = 1):
    # Sound Event Detection 추천 feature 사용
    spectrogram = T.MelSpectrogram(sample_rate= sample_rate,
                                hop_length = hop_length,
                                n_fft = n_fft,
                                n_mels = n_mels,
                                window_fn = window_fn,
                                win_length = win_length,
                                power = power) # 1 : energy, 2 : power
    return(spectrogram)


