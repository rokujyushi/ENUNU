"""モデルを使わない単体テスト (設定の解決・キャッシュの入出力・ピッチ差し替え・拡張機能の実行)。"""
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import _common  # noqa: F401  (sys.path の設定)
import enunu
import enunu_server as S
from enulib.extensions import run_extension

DIFFUSION_ENV = ('ENUNU_DIFFUSION_SPEEDUP', 'ENUNU_DIFFUSION_TARGETS', 'ENUNU_DIFFUSION_METHOD',
                 'ENUNU_DIFFUSION_MGC', 'ENUNU_DIFFUSION_MEL', 'ENUNU_DIFFUSION_BAP')


def clean_env(**values):
    env = {k: v for k, v in os.environ.items() if k not in DIFFUSION_ENV}
    env.update(values)
    return mock.patch.dict(os.environ, env, clear=True)


def brief(settings):
    return {k: f"{v['method']}:{v['steps']}" for k, v in settings.items()}


class TestParseStyleShift(unittest.TestCase):
    def test_values(self):
        base = ['acoustic', 'a.tmp', '', 'hash', '600']
        self.assertEqual(S.parse_style_shift(base, 5), 0)              # 旧クライアント (5要素)
        self.assertEqual(S.parse_style_shift(base + ['3'], 5), 3)
        self.assertEqual(S.parse_style_shift(base + [-2], 5), -2)
        self.assertEqual(S.parse_style_shift(base + [1.6], 5), 2)
        for bad in (None, 'abc', True, [1.0], {'a': 1}):
            self.assertEqual(S.parse_style_shift(base + [bad], 5), 0, bad)
        # acoustic_f0 は [5] が f0 配列、[6] が style_shift
        self.assertEqual(S.parse_style_shift(base + [[100.0, 0.0], 4], 6), 4)


class TestDiffusionSettings(unittest.TestCase):
    def tearDown(self):
        enunu.set_diffusion_settings({'reset': True})

    def test_default(self):
        with clean_env():
            self.assertEqual(brief(enunu.diffusion_settings()),
                             {'mgc': 'ddim:25', 'mel': 'ddim:25', 'bap': 'plms:20', 'other': 'ddpm:100'})

    def test_legacy_env(self):
        with clean_env(ENUNU_DIFFUSION_TARGETS='all', ENUNU_DIFFUSION_METHOD='eta1'):
            self.assertEqual(brief(enunu.diffusion_settings()),
                             {'mgc': 'eta1:10', 'mel': 'eta1:10', 'bap': 'eta1:10', 'other': 'ddpm:100'})
        with clean_env(ENUNU_DIFFUSION_SPEEDUP='1'):
            self.assertTrue(all(v['method'] == 'ddpm' for v in enunu.diffusion_settings().values()))
        with clean_env(ENUNU_DIFFUSION_SPEEDUP='5'):  # TARGETS 省略時は従来どおり bap のみ
            self.assertEqual(brief(enunu.diffusion_settings())['bap'], 'plms:20')
            self.assertEqual(brief(enunu.diffusion_settings())['mgc'], 'ddpm:100')

    def test_stream_env(self):
        with clean_env(ENUNU_DIFFUSION_MEL='ddim:20', ENUNU_DIFFUSION_MGC='50', ENUNU_DIFFUSION_BAP='foo:3'):
            b = brief(enunu.diffusion_settings())
            self.assertEqual(b['mel'], 'ddim:20')
            self.assertEqual(b['mgc'], 'ddim:50')   # 数字だけならステップ数だけ変える
            self.assertEqual(b['bap'], 'plms:20')   # 不正値は無視

    def test_config(self):
        with clean_env():
            self.assertEqual(brief(enunu.set_diffusion_settings({'steps': 10}))['mgc'], 'ddim:10')
            self.assertEqual(brief(enunu.diffusion_settings())['mel'], 'ddim:10')
            enunu.set_diffusion_settings({'bap': 'ddpm', 'mgc': {'method': 'eta1'}})
            b = brief(enunu.diffusion_settings())
            self.assertEqual((b['mgc'], b['bap'][:4]), ('eta1:10', 'ddpm'))
            for bad in ({'steps': 0}, {'steps': 'x'}, {'steps': True}, {'mgc': 'foo:10'}):
                with self.assertRaises((ValueError, TypeError), msg=bad):
                    enunu.set_diffusion_settings(bad)
            self.assertEqual(brief(enunu.diffusion_settings())['mgc'], 'eta1:10')  # 不正値では変わらない
            enunu.set_diffusion_settings({'reset': True})
            self.assertEqual(brief(enunu.diffusion_settings())['mgc'], 'ddim:25')


def fake_engine(tmp, feature_type='world'):
    e = types.SimpleNamespace()
    e.feature_type = feature_type
    e.path_features_npz = os.path.join(tmp, 'features.npz')
    e.path_editorf0_npy = os.path.join(tmp, 'editorf0.npy')
    e.path_f0_npy = os.path.join(tmp, 'f0.npy')
    e.path_mel_npy = os.path.join(tmp, 'mel.npy')
    e.path_vuv_npy = os.path.join(tmp, 'vuv.npy')
    return e


class TestFeatureCache(unittest.TestCase):
    def test_roundtrip_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = fake_engine(tmp, 'world')
            rng = np.random.default_rng(0)
            e.multistream_features = (rng.normal(size=(50, 60)), rng.normal(size=(50, 1)),
                                      rng.uniform(size=(50, 1)), rng.normal(size=(50, 5)))
            S.save_features(e, 'acoustic', 2, 'digest')
            features, meta = S.load_features(e)
            self.assertTrue(all(np.array_equal(a, b) for a, b in zip(e.multistream_features, features)))
            self.assertEqual(meta, S.features_meta(e, 'acoustic', 2, 'digest'))
            e.feature_type = 'melf0'   # 別形式のキャッシュは使わない
            self.assertEqual(S.load_features(e), (None, None))

    def test_broken_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = fake_engine(tmp)
            with open(e.path_features_npz, 'wb') as f:
                f.write(b'not a npz')
            self.assertEqual(S.load_features(e), (None, None))

    def test_legacy_melf0(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = fake_engine(tmp, 'melf0')
            f0 = np.array([0, 0, 200, 0, 400, 0], dtype=np.float64)
            np.save(e.path_f0_npy, f0)
            np.save(e.path_mel_npy, np.zeros((6, 80)))
            np.save(e.path_vuv_npy, (f0 > 0).astype(np.float64))
            mel, lf0, vuv = S.load_legacy_melf0(e)
            self.assertEqual(lf0.shape, (6, 1))
            # 無声区間は補間された連続値 (端は最近傍)
            np.testing.assert_allclose(np.exp(lf0.flatten()), [200, 200, 200, np.sqrt(200 * 400), 400, 400])


class TestEditorF0(unittest.TestCase):
    def test_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = fake_engine(tmp)
            lf0 = np.log(np.full((5, 1), 220.0)).astype(np.float32)
            features = (np.zeros((5, 3)), lf0, np.ones((5, 1)))
            self.assertIs(S.apply_editor_f0(e, features), features)   # ファイルが無ければそのまま
            np.save(e.path_editorf0_npy, np.array([0, 440, 440, 0, 110], dtype=np.float64))
            out = S.apply_editor_f0(e, features)
            np.testing.assert_allclose(np.exp(out[1].flatten()), [220, 440, 440, 220, 110], rtol=1e-5)
            self.assertEqual(out[1].dtype, np.float32)
            # フレーム数が違うときは重なる範囲だけ
            np.save(e.path_editorf0_npy, np.array([330, 330], dtype=np.float64))
            out = S.apply_editor_f0(e, features)
            np.testing.assert_allclose(np.exp(out[1].flatten()), [330, 330, 220, 220, 220], rtol=1e-5)


class TestNnsvsSpeedups(unittest.TestCase):
    """nnsvs_speedups の差し替えが元の関数と同じ結果を返すこと。"""

    def setUp(self):
        from enulib import nnsvs_speedups
        from nnmnkwii.frontend import merlin
        from nnmnkwii.io import hts
        nnsvs_speedups.apply()
        nnsvs_speedups.clear_caches()
        self.merlin = merlin
        with tempfile.NamedTemporaryFile('w', suffix='.hed', delete=False, encoding='utf-8') as f:
            f.write('QS "C-Phone_a" {*-a+*}\nQS "C-Phone_Vowel" {*-a+*,*-i+*,*-u+*}\n'
                    'QS "L-Phone_pau" {pau-*}\nCQS "C-Pos" {/A:(\\d+)_}\n')
            self.hed = f.name
        self.binary_dict, self.numeric_dict = hts.load_question_set(self.hed)
        lines = ['0 500000 xx^pau-a+i/A:1_', '500000 1500000 pau^a-i+u/A:2_', '1500000 2500000 a^i-u+pau/A:3_',
                 '2500000 3000000 i^u-pau+xx/A:4_']
        with tempfile.NamedTemporaryFile('w', suffix='.lab', delete=False, encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
            self.lab = f.name
        self.labels = hts.load(self.lab)

    def tearDown(self):
        os.remove(self.hed)
        os.remove(self.lab)

    def test_same_as_original(self):
        m = self.merlin
        for label in self.labels.contexts:
            for _ in range(2):   # 2回目はキャッシュから
                np.testing.assert_array_equal(m.pattern_matching_binary(self.binary_dict, label),
                                              m.pattern_matching_binary.__wrapped__(self.binary_dict, label))
                np.testing.assert_array_equal(m.pattern_matching_continous_position(self.numeric_dict, label),
                                              m.pattern_matching_continous_position.__wrapped__(self.numeric_dict, label))
        for kwargs in ({'add_frame_features': True, 'frame_shift': 50000},
                       {'add_frame_features': True, 'frame_shift': 50000, 'subphone_features': 'coarse_coding'},
                       {'add_frame_features': False}):
            expected = m.linguistic_features.__wrapped__(self.labels, self.binary_dict, self.numeric_dict, **kwargs)
            first = m.linguistic_features(self.labels, self.binary_dict, self.numeric_dict, **kwargs)
            first[:] = -1   # 呼び出し側が書き換えてもキャッシュは壊れない
            second = m.linguistic_features(self.labels, self.binary_dict, self.numeric_dict, **kwargs)
            np.testing.assert_array_equal(second, expected)

    def test_single_sequence_lstm(self):
        import torch
        from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            torch.manual_seed(0)
            lstm = torch.nn.LSTM(8, 16, 2, batch_first=True, bidirectional=True).to(device).eval()
            x = torch.randn(1, 50, 8, device=device)
            with torch.no_grad():
                packed_out, (h, c) = lstm(pack_padded_sequence(x, [50], batch_first=True))
                self.assertIsInstance(packed_out, PackedSequence)
                out, _ = pad_packed_sequence(packed_out, batch_first=True)
                ref, (h_ref, c_ref) = lstm(x)   # 通常のテンソル (差し替えの対象外)
                torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(h, h_ref, rtol=1e-5, atol=1e-5)
                # バッチが 2 以上なら元の経路 (パックのまま)
                x2 = torch.randn(2, 50, 8, device=device)
                out2, _ = lstm(pack_padded_sequence(x2, [50, 30], batch_first=True))
                self.assertIsInstance(out2, PackedSequence)
                self.assertEqual(out2.data.shape[0], 80)

    def test_mc2sp_vectorized(self):
        import pysptk
        import pysptk.conversion
        original = pysptk.conversion.mc2sp.__wrapped__   # apply() で差し替えた関数の元
        rng = np.random.default_rng(1)
        for dims, fftlen, alpha in ((60, 2048, 0.58), (15, 2048, 0.58), (25, 1024, 0.466)):
            mc = rng.normal(scale=0.3, size=(40, dims))
            np.testing.assert_allclose(pysptk.mc2sp(mc, alpha, fftlen), original(mc, alpha, fftlen), rtol=1e-10)
            np.testing.assert_allclose(pysptk.mc2sp(mc[0], alpha, fftlen), original(mc[0], alpha, fftlen), rtol=1e-10)

    @unittest.skipUnless(__import__('torch').cuda.is_available(), 'CUDA が必要')
    def test_nonattentive_decoder_graph_matches_eager(self):
        """自己回帰デコーダーのグラフ版が、同じシードで通常版と完全に一致すること。

        prenet が無い場合は (1, out_dim, rf) のビューに dropout をかけるので、メモリ配置まで合わせないと
        乱数の割り当てが変わる (rf=2, out_dim=60 で不一致になった不具合の再発防止)。
        """
        import torch
        from enulib import nnsvs_speedups
        from nnsvs.tacotron.decoder import NonAttentiveDecoder
        nnsvs_speedups.apply_cuda_graphs()
        for prenet_layers in (0, 2):
            for rf in (1, 2):
                for out_dim in (1, 5, 60):
                    torch.manual_seed(0)
                    dec = NonAttentiveDecoder(in_dim=32, out_dim=out_dim, layers=2, hidden_dim=64,
                                              prenet_layers=prenet_layers, prenet_hidden_dim=16,
                                              reduction_factor=rf, downsample_by_conv=True).cuda().eval()
                    x = torch.randn(1, 40 * rf, 32, device='cuda')
                    with torch.no_grad():
                        dec._enunu_graph_failed = True
                        torch.manual_seed(3)
                        eager = dec(x, [40 * rf])
                        dec._enunu_graph_failed = False
                        torch.manual_seed(3)
                        graphed = dec(x, [40 * rf])
                    self.assertIsNotNone(getattr(dec, '_enunu_step_graph', None))
                    torch.testing.assert_close(graphed, eager, rtol=0, atol=0,
                                               msg=f'prenet={prenet_layers} rf={rf} out_dim={out_dim}')

    def test_mcepalpha_cached(self):
        import pysptk.util
        self.assertEqual(pysptk.util.mcepalpha(48000), pysptk.util.mcepalpha.__wrapped__(48000))
        self.assertGreater(pysptk.util.mcepalpha.cache_info().hits + 1, 0)


@unittest.skipUnless(__import__('torch').cuda.is_available(), 'CUDA が必要')
class TestGraphedDenoiser(unittest.TestCase):
    def test_same_output_as_eager(self):
        import torch
        from nnsvs.diffsinger.denoiser import DiffNet
        torch.manual_seed(0)
        net = DiffNet(in_dim=20, encoder_hidden_dim=32, residual_layers=4, residual_channels=32).cuda().eval()
        torch.nn.init.normal_(net.output_projection.weight)   # 既定はゼロ初期化で出力が常に 0 になるため
        graphed = enunu.GraphedDenoiser(net, max_graphs=2)
        with torch.no_grad():
            for T in (50, 80, 50, 120):          # 長さが変わる・戻る・上限を超える
                x = torch.randn(1, 1, 20, T, device='cuda')
                t = torch.tensor([7], device='cuda')
                cond = torch.randn(1, 32, T, device='cuda')
                torch.testing.assert_close(graphed(x, t, cond), net(x, t, cond=cond), rtol=0, atol=0)
        self.assertEqual(len(graphed.graphs), 2)  # LRU で上限を守る
        self.assertEqual(graphed.failed, set())
        # 勾配を計算するときはグラフを使わない
        x.requires_grad_(True)
        self.assertTrue(graphed(x, t, cond).requires_grad)


class TestEditAcousticReload(unittest.TestCase):
    """edit_acoustic: 拡張機能が書き換えなかった CSV は読み直さず、元の配列 (丸めなし) を使う。"""

    def run_edit(self, script):
        with tempfile.TemporaryDirectory() as tmp:
            ext = os.path.join(tmp, 'ext.py')
            with open(ext, 'w', encoding='utf-8') as f:
                f.write(script)
            fake = types.SimpleNamespace(
                path_mgc=os.path.join(tmp, 'mgc.csv'), path_f0=os.path.join(tmp, 'f0.csv'),
                path_vuv=os.path.join(tmp, 'vuv.csv'), path_bap=os.path.join(tmp, 'bap.csv'),
                path_ust=None, path_table=None, path_feedback=None, path_full_score=None,
                path_mono_score=None, path_full_timing=None, path_mono_timing=None,
                get_extension_path_list=lambda key: [ext])
            rng = np.random.default_rng(0)
            features = (rng.normal(size=(20, 6)), np.log(rng.uniform(100, 400, size=(20, 1))),
                        rng.uniform(size=(20, 1)), rng.normal(size=(20, 3)))
            out = enunu.ENUNU.edit_acoustic(fake, features, 'world')
            self.assertFalse(os.path.exists(fake.path_mgc))   # CSV は後片付けされる
            return features, out

    def test_only_changed_files_are_reloaded(self):
        # f0 だけを 2 倍にする拡張機能 (style_shifter と同じく f0 だけを書き換える)
        features, out = self.run_edit(
            'import sys, numpy as np\nargs = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n'
            'f0 = np.loadtxt(args["--f0"], delimiter=",")\nnp.savetxt(args["--f0"], f0 * 2, fmt="%.9g", delimiter=",")\n')
        np.testing.assert_array_equal(out[0], features[0])          # mgc は元の配列そのもの
        np.testing.assert_array_equal(out[3], features[3])          # bap も
        np.testing.assert_allclose(out[1], features[1] + np.log(2), atol=1e-7)   # f0 は読み直し

    def test_unchanged_everything(self):
        features, out = self.run_edit('pass\n')
        for a, b in zip(features, out):
            np.testing.assert_array_equal(a, b)

    def test_changed_mgc_is_reloaded(self):
        features, out = self.run_edit(
            'import sys, numpy as np\nargs = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n'
            'np.savetxt(args["--mgc"], np.zeros((20, 6)), fmt="%.9g", delimiter=",")\n')
        np.testing.assert_array_equal(out[0], np.zeros((20, 6)))
        np.testing.assert_array_equal(out[1], features[1])


class TestExtensionInProcess(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        for v in ('v1', 'v2'):
            os.makedirs(os.path.join(root, v))
            self.write(f'{v}/helper.py', f'X = "{v}"\n')
            self.write(f'{v}/ext.py', 'import sys, os, helper\nif __name__ == "__main__":\n'
                                      '    with open(sys.argv[2], "w") as f:\n'
                                      '        f.write(helper.X + "|" + os.path.basename(os.getcwd()))\n')
        # 同じフォルダのモジュールを使わない拡張機能 (subprocess 実行でも動く)
        self.write('v1/plain.py', 'import sys, os\nif __name__ == "__main__":\n'
                                  '    with open(sys.argv[2], "w") as f:\n'
                                  '        f.write(sys.argv[1] + "|" + os.path.basename(os.getcwd()))\n')
        self.write('v1/exit0.py', 'import sys\nsys.exit(0)\n')
        self.write('v1/exit3.py', 'import sys\nsys.exit(3)\n')
        self.write('v1/boom.py', 'raise RuntimeError("boom")\n')
        self.write('v1/imp.py', 'import not_a_module_xyz\n')

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rel, text):
        with open(os.path.join(self.tmp.name, rel), 'w', encoding='utf-8') as f:
            f.write(text)

    def path(self, rel):
        return os.path.join(self.tmp.name, rel)

    def read(self, rel):
        with open(self.path(rel)) as f:
            return f.read()

    def test_argv_cwd_and_module_isolation(self):
        cwd = os.getcwd()
        run_extension(self.path('v1/ext.py'), out=self.path('out1.txt'), unused=None)
        run_extension(self.path('v2/ext.py'), out=self.path('out2.txt'))
        self.assertEqual(self.read('out1.txt'), 'v1|v1')   # cwd は拡張機能のフォルダ
        self.assertEqual(self.read('out2.txt'), 'v2|v2')   # 同名の helper が音源ごとに別物になる
        self.assertEqual(os.getcwd(), cwd)
        self.assertNotIn('helper', sys.modules)

    def test_exit_codes(self):
        run_extension(self.path('v1/exit0.py'))
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            run_extension(self.path('v1/exit3.py'))
        self.assertEqual(cm.exception.returncode, 3)
        with self.assertRaises(subprocess.CalledProcessError):
            run_extension(self.path('v1/boom.py'))

    def test_import_error_falls_back_to_subprocess(self):
        with mock.patch('subprocess.run', wraps=subprocess.run) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                run_extension(self.path('v1/imp.py'))
            self.assertEqual(run.call_count, 1)

    def test_same_result_as_subprocess(self):
        run_extension(self.path('v1/plain.py'), out=self.path('in.txt'))
        with mock.patch.dict(os.environ, {'ENUNU_EXTENSION_INPROCESS': '0'}), \
                mock.patch('subprocess.run', wraps=subprocess.run) as run:
            run_extension(self.path('v1/plain.py'), out=self.path('sub.txt'))
            self.assertEqual(run.call_count, 1)
        self.assertEqual(self.read('in.txt'), '--out|v1')
        self.assertEqual(self.read('sub.txt'), self.read('in.txt'))


if __name__ == '__main__':
    unittest.main()
