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
