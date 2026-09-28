"""nnsvs のモデルを ONNX に書き出せる形にするラッパー。

書き出しの対象は「バッチ 1・全フレーム有効」で動かす推論だけ。nnsvs のモデルには、そのままだと
ONNX にできない箇所が2つある。

- LSTM に渡す前後の pack_padded_sequence / pad_packed_sequence
  (バッチ 1 で全フレームが有効なら何もしなくてよいので、書き出しの間だけ素通しにする)
- MDN の「最も重みの大きい成分の mu, sigma を取る」処理 (mdn_get_most_probable_sigma_and_mu)
  (to_one_hot が torch.FloatTensor で定数を作るので、gather で書き直したものに差し替える)
"""
import contextlib
import sys
from unittest import mock

import torch
from torch import nn

from nnsvs.base import PredictionType


def most_probable_sigma_and_mu(log_pi, log_sigma, mu):
    """nnsvs.mdn.mdn_get_most_probable_sigma_and_mu と同じ結果を返す、ONNX にできる実装。"""
    dim_wise = log_pi.dim() == 4
    # (B, T, 1) か (B, T, 1, D)
    idx = torch.argmax(log_pi, dim=2, keepdim=True)
    if not dim_wise:
        idx = idx.unsqueeze(-1).expand(-1, -1, -1, mu.shape[-1])
    max_mu = torch.gather(mu, 2, idx).squeeze(2)
    max_sigma = torch.exp(torch.gather(log_sigma, 2, idx).squeeze(2))
    return max_sigma, max_mu


def _passthrough_pack(x, lengths=None, batch_first=False, enforce_sorted=True):
    return x


def _passthrough_pad(sequence, batch_first=False, padding_value=0.0, total_length=None):
    return sequence, None


@contextlib.contextmanager
def export_patches():
    """nnsvs 内の該当関数を、import 先の全モジュールで差し替える。"""
    replacements = {
        'pack_padded_sequence': _passthrough_pack,
        'pad_packed_sequence': _passthrough_pad,
        'mdn_get_most_probable_sigma_and_mu': most_probable_sigma_and_mu,
    }
    patches = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith('nnsvs'):
            continue
        for attr, func in replacements.items():
            if hasattr(module, attr):
                patches.append(mock.patch.object(module, attr, func))
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


class InferenceWrapper(nn.Module):
    """model.inference(x) を、テンソルだけを返す forward にする。

    出力は、確率モデル (MDN) なら (mu, sigma)、そうでなければ (y,)。
    どちらも nnsvs の推論 (SPSVS / ENUNU) が使う値と同じ。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.probabilistic = model.prediction_type() == PredictionType.PROBABILISTIC

    @property
    def output_names(self):
        return ['mu', 'sigma'] if self.probabilistic else ['y']

    def forward(self, x):
        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long)
        out = self.model.inference(x, lengths)
        if self.probabilistic:
            mu, sigma = out
            return mu, sigma
        return out
