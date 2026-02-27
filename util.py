import torch
import torchaudio

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(s: str) -> str:
    '''
    规范化文本：去首尾空白 + 压缩连续空白
    
    Args:
        s: 文本
    Returns:
        s: 规范化后的文本
    '''
    s = s.strip()
    s = " ".join(s.split())
    return s


def load_audio_16k_mono(path: str) -> torch.Tensor:
    '''
    Args:
        path: 音频文件路径
    Returns:
        wav: 16k、单声道 的 1D waveform
    '''

    wav, sr = torchaudio.load(path)  # (C, T)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)  # (T,)