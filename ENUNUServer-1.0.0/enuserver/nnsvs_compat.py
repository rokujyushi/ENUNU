"""NumPy 2 (Python 3.13) で動かない nnsvs の関数を、直した版に差し替える。

taroushirani/nnsvs の python3.13-support ブランチと同じ修正。NumPy 1.26 でも結果は変わらない。

- nnsvs.dsp.lowpass_filter: scipy.signal.butter に要素1個のリスト (Wn=[x]) を渡していた。
  NumPy 2 + SciPy 1.18 では TypeError になり、trajectory_smoothing (既定で有効) で必ず止まる。
"""
from scipy import signal

_applied = False


def lowpass_filter(x, fs, cutoff=5, N=5):
    """nnsvs.dsp.lowpass_filter と同じ。Wn をスカラーで渡すところだけ違う。"""
    nyquist = fs // 2
    b, a = signal.butter(N, cutoff / nyquist, 'lowpass')
    if len(x) <= max(len(a), len(b)) * (N // 2 + 1):
        # 信号が短すぎる
        return x
    # ゼロ位相フィルタ
    return signal.filtfilt(b, a, x)


def apply():
    """差し替えを適用する (何度呼んでもよい)。"""
    global _applied
    if _applied:
        return
    _applied = True

    import nnsvs.dsp
    import nnsvs.gen
    import nnsvs.pitch

    # nnsvs.pitch と nnsvs.gen は `from ... import lowpass_filter` で取り込んでいるので、それぞれ差し替える
    for module in (nnsvs.dsp, nnsvs.pitch, nnsvs.gen):
        module.lowpass_filter = lowpass_filter
