"""実モデルを使う通しテスト (サーバーの各コマンドを、リクエストと同じ手順でプロセス内から呼ぶ)。

音源が見つからなければスキップする。1音源あたり 30〜60 秒程度かかる。
"""
import itertools
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import _common
import enunu
import enunu_server as S

VOICES = _common.test_voices()


class Counter:
    """enunu.run_acoustic / run_pitch の呼び出し回数 (= 推論したかどうか) を数える。"""

    def __init__(self):
        self.acoustic = 0
        self.pitch = 0
        orig_acoustic, orig_pitch = enunu.run_acoustic, enunu.run_pitch

        def acoustic(*a, **k):
            self.acoustic += 1
            return orig_acoustic(*a, **k)

        def pitch(*a, **k):
            self.pitch += 1
            return orig_pitch(*a, **k)
        self.patches = [mock.patch.object(enunu, 'run_acoustic', acoustic),
                        mock.patch.object(enunu, 'run_pitch', pitch)]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def median_cents(wav_a, wav_b):
    """2つの wav の F0 の差 (cent, 中央値)。"""
    import pyworld
    from scipy.io import wavfile
    f0s = []
    for path in (wav_a, wav_b):
        sr, x = wavfile.read(path)
        f0, _ = pyworld.harvest(x.astype(np.float64), sr, frame_period=5)
        f0s.append(f0)
    n = min(len(f0s[0]), len(f0s[1]))
    a, b = f0s[0][:n], f0s[1][:n]
    voiced = (a > 0) & (b > 0)
    return float(np.median(1200 * np.log2(b[voiced] / a[voiced])))


class ServerE2EBase:
    voice = None

    @classmethod
    def setUpClass(cls):
        cls.cwd = os.getcwd()
        cls.work = tempfile.mkdtemp(prefix='enunu-test-')
        cls.tmp_a = _common.write_tmp(os.path.join(cls.work, 'enu-a.tmp'), cls.voice, _common.PHRASE_A)
        cls.tmp_b = _common.write_tmp(os.path.join(cls.work, 'enu-b.tmp'), cls.voice, _common.PHRASE_B)
        cls.engine = enunu.setup(cls.tmp_a)
        cls.wav_ids = itertools.count()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.cwd)
        enunu.set_diffusion_settings({'reset': True})
        shutil.rmtree(cls.work, ignore_errors=True)

    def setUp(self):
        # テストごとにワークフォルダを空にする
        for name in os.listdir(self.work):
            if name.endswith('_enutemp'):
                shutil.rmtree(os.path.join(self.work, name))

    def request(self, command, tmp, *args):
        """サーバーのメインループと同じく、update_path してからコマンドを呼ぶ。"""
        enunu.update_path(tmp, self.engine)
        if command == 'acoustic':
            return S.acoustic(self.engine, *args)
        if command == 'pitch':
            return S.pitch(self.engine, *args)
        if command == 'acoustic_f0':
            return S.acoustic_f0(self.engine, *args)
        if command == 'synthe':
            wav = os.path.join(self.work, f'out{next(self.wav_ids)}.wav')
            S.synthe(wav, self.engine, *args)
            return wav
        raise ValueError(command)

    def test_sp_ap_only_when_client_reads_them(self):
        # OpenUtau が synthe で合成する音源 (melf0 や wav_synthesizer: synthe) では sp/ap の npy を作らない
        r = self.request('acoustic', self.tmp_a)
        reads = self.engine.client_reads_world_params()
        self.assertTrue(os.path.isfile(r['path_f0']))
        self.assertEqual(os.path.isfile(r['path_spectrogram']), reads)
        self.assertEqual(os.path.isfile(r['path_aperiodicity']), reads)

    def test_phrase_switch_uses_per_phrase_cache(self):
        with Counter() as c:
            ra = self.request('acoustic', self.tmp_a)
            rb = self.request('acoustic', self.tmp_b)
            self.assertEqual(c.acoustic, 2)   # B は A のキャッシュを使わない
            self.assertNotEqual(os.path.dirname(ra['path_f0']), os.path.dirname(rb['path_f0']))
            self.request('acoustic', self.tmp_a)
            self.request('acoustic', self.tmp_b)
            self.assertEqual(c.acoustic, 2)   # 2回目はどちらもキャッシュ
            # メモリには B の特徴量があるが、A の synthe はディスクのキャッシュから推論なしで合成する
            wav = self.request('synthe', self.tmp_a)
            self.assertEqual(c.acoustic, 2)
            self.assertTrue(os.path.getsize(wav) > 1000)

    def test_synthe_after_restart_and_editorf0(self):
        with Counter() as c:
            r = self.request('acoustic', self.tmp_a)
            wav_model = self.request('synthe', self.tmp_a)
            self.engine.multistream_features = None   # 再起動相当
            f0 = np.load(r['path_f0'])
            np.save(self.engine.path_editorf0_npy, np.where(f0 > 0, f0 * 2 ** (200 / 1200), 0))
            try:
                wav_edit = self.request('synthe', self.tmp_a)
            finally:
                os.remove(self.engine.path_editorf0_npy)
            self.assertEqual(c.acoustic, 1)
        self.assertAlmostEqual(median_cents(wav_model, wav_edit), 200, delta=15)

    def test_ust_change_invalidates(self):
        with Counter() as c:
            self.request('acoustic', self.tmp_a)
            _common.write_tmp(self.tmp_a, self.voice, _common.PHRASE_A[:2] + [('う', 960, 65)])
            try:
                self.request('acoustic', self.tmp_a)
            finally:
                _common.write_tmp(self.tmp_a, self.voice, _common.PHRASE_A)
            self.assertEqual(c.acoustic, 2)

    def test_style_shift_and_config_invalidate(self):
        with Counter() as c:
            self.request('acoustic', self.tmp_a)
            self.request('acoustic', self.tmp_a, 2)
            self.assertEqual(c.acoustic, 2)
            S.config({'diffusion': {'steps': 10}}, {'x': (self.engine, 600, 0)})
            try:
                self.request('acoustic', self.tmp_a, 2)
                self.assertEqual(c.acoustic, 3)
            finally:
                S.config({'diffusion': {'reset': True}}, {'x': (self.engine, 600, 0)})

    def test_pitch_and_acoustic_f0_cache(self):
        with Counter() as c:
            r = self.request('pitch', self.tmp_a)
            f0 = np.load(r['path_f0'])
            self.assertEqual(r['n_frames'], len(f0))
            self.assertTrue(np.all(f0 >= 0) and np.any(f0 > 0))
            r2 = self.request('pitch', self.tmp_a)
            self.assertTrue(np.array_equal(f0, np.load(r2['path_f0'])))
            self.request('pitch', self.tmp_a, 3)
            self.assertEqual(c.pitch, 2)           # 2回目はキャッシュ、style_shift が変われば計算

            editor = f0 * 2 ** (100 / 1200)
            self.request('acoustic_f0', self.tmp_a, editor)
            n = c.acoustic
            self.request('acoustic_f0', self.tmp_a, editor)
            self.assertEqual(c.acoustic, n)        # 同じ入力はキャッシュ
            editor[len(editor) // 2] += 5.0
            self.request('acoustic_f0', self.tmp_a, editor)
            self.assertEqual(c.acoustic, n + 1)    # 1フレームでも違えば計算
            self.request('acoustic', self.tmp_a)
            self.assertEqual(c.acoustic, n + 2)    # acoustic_f0 の結果は acoustic のキャッシュにならない

    def clear_work_caches(self):
        for path in (self.engine.path_features_npz, self.engine.path_pitch_npy, self.engine.path_pitch_lf0_npy,
                     S.pitch_meta_path(self.engine)):
            if os.path.isfile(path):
                os.remove(path)
        self.engine.multistream_features = None

    def test_reproducible_with_fixed_seed(self):
        self.request('acoustic', self.tmp_a)
        first = [np.array(f) for f in self.engine.multistream_features]
        wav1 = self.request('synthe', self.tmp_a)
        self.clear_work_caches()
        with Counter() as c:
            self.request('acoustic', self.tmp_a)
            self.assertEqual(c.acoustic, 1)
        second = self.engine.multistream_features
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, np.asarray(b))
        wav2 = self.request('synthe', self.tmp_a)
        from scipy.io import wavfile
        np.testing.assert_array_equal(wavfile.read(wav1)[1], wavfile.read(wav2)[1])

    def test_acoustic_reuses_lf0_from_pitch(self):
        if not self.engine.supports_lf0_conditioning():
            self.skipTest('lf0_model を持つモデルのみ')
        lf0_cls = type(self.engine.acoustic_model.lf0_model)
        calls = []
        orig = lf0_cls.inference

        def counting(model, *a, **k):
            calls.append(1)
            return orig(model, *a, **k)
        with mock.patch.object(lf0_cls, 'inference', counting):
            self.request('acoustic', self.tmp_a)            # acoustic 単体
            alone = [np.array(f) for f in self.engine.multistream_features]
            self.assertEqual(len(calls), 1)
            self.clear_work_caches()
            self.request('pitch', self.tmp_a)               # pitch → acoustic
            self.request('acoustic', self.tmp_a)
            self.assertEqual(len(calls), 2)                 # acoustic では lf0_model を実行しない
            after_pitch = self.engine.multistream_features
            f0 = np.load(self.engine.path_f0_npy)
            self.request('acoustic_f0', self.tmp_a, f0)     # acoustic_f0 でも実行しない
            self.assertEqual(len(calls), 2)
        # lf0 を使い回しても、シード固定により acoustic 単体と同じ結果になる
        self.assertTrue(all(np.array_equal(a, np.asarray(b)) for a, b in zip(alone, after_pitch)))

    def test_lf0_decoder_graph_matches_eager(self):
        import torch
        from nnsvs.acoustic_models.tacotron_f0 import ResF0NonAttentiveDecoder
        decoder = getattr(getattr(self.engine.acoustic_model, 'lf0_model', None), 'decoder', None)
        if not (isinstance(decoder, ResF0NonAttentiveDecoder) and torch.cuda.is_available()):
            self.skipTest('ResF0NonAttentiveDecoder と CUDA が必要')
        from nnmnkwii.io import hts
        enunu.update_path(self.tmp_a, self.engine)
        enunu.run_timing(engine=self.engine, step='acoustic')
        labels = hts.load(self.engine.path_full_timing).round_()
        labels.frame_shift = int(self.engine.config.frame_period * 1e4)
        results = []
        for eager in (False, True):
            decoder._enunu_graph_failed = eager   # True なら元の forward を使う
            try:
                torch.manual_seed(11)
                results.append(self.engine.predict_lf0(labels)[0])
            finally:
                decoder._enunu_graph_failed = False
        self.assertIsNotNone(getattr(decoder, '_enunu_step_graph', None))   # グラフ版が使われた
        np.testing.assert_array_equal(results[0], results[1])

    def test_synthe_without_tmp(self):
        # OpenUtau の「選択ノートのキャッシュ削除」は tmp だけを消し、_enutemp は残す
        self.request('acoustic', self.tmp_a)
        wav_normal = self.request('synthe', self.tmp_a)
        backup = self.tmp_a + '.bak'
        os.replace(self.tmp_a, backup)
        try:
            with Counter() as c:
                path, cache_only = S.resolve_plugin_path(['synthe', self.tmp_a, '', 'X', '600'])
                self.assertTrue(cache_only)
                self.assertTrue(path.endswith('temp.ust'))
                enunu.update_path(path, self.engine)
                wav = os.path.join(self.work, f'out{next(self.wav_ids)}.wav')
                S.synthe(wav, self.engine, 0, cache_only)
                self.assertEqual(c.acoustic, 0)
            from scipy.io import wavfile
            np.testing.assert_array_equal(wavfile.read(wav)[1], wavfile.read(wav_normal)[1])
            # キャッシュも無ければ分かりやすいエラー (melf0 は旧形式の mel.npy からも復元するので消す)
            os.remove(self.engine.path_features_npz)
            if os.path.isfile(self.engine.path_mel_npy):
                os.remove(self.engine.path_mel_npy)
            with self.assertRaises(FileNotFoundError):
                S.synthe(wav, self.engine, 0, cache_only)
        finally:
            os.replace(backup, self.tmp_a)
        # tmp もワークフォルダも無い場合
        with self.assertRaises(FileNotFoundError):
            S.resolve_plugin_path(['synthe', os.path.join(self.work, 'enu-none.tmp'), '', 'X', '600'])
        # synthe 以外は従来どおり
        self.assertEqual(S.resolve_plugin_path(['acoustic', self.tmp_a, '', 'X', '600']), (self.tmp_a, False))

    def test_legacy_melf0_workfolder(self):
        if self.engine.feature_type != 'melf0':
            self.skipTest('melf0 モデルのみ')
        with Counter() as c:
            self.request('acoustic', self.tmp_a)
            os.remove(self.engine.path_features_npz)   # 旧バージョンのワークフォルダ相当
            self.engine.multistream_features = None
            wav = self.request('synthe', self.tmp_a)
            self.assertEqual(c.acoustic, 1)
            self.assertTrue(os.path.getsize(wav) > 1000)


# 見つかった音源ごとにテストクラスを作る
for _i, _voice in enumerate(VOICES):
    _name = f'TestServerE2E_{_i}_{os.path.basename(_voice).encode("ascii", "ignore").decode() or _i}'
    globals()[_name] = type(_name, (ServerE2EBase, unittest.TestCase), {'voice': _voice})

if not VOICES:
    class TestServerE2E(unittest.TestCase):
        def test_skipped(self):
            self.skipTest('テスト用の音源が見つからない (ENUNU_TEST_VOICES で指定)')

if __name__ == '__main__':
    unittest.main()
