"""サーバープロセスを起動し、OpenUtau と同じ形式のリクエストを ZMQ で送る通しテスト。

ENUNU_SERVER_PORT で 15599 番を使うので、OpenUtau が使う 15556 番とはぶつからない。
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest

import zmq

import _common

PORT = '15599'


class TestZmqProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.mkdtemp(prefix='enunu-zmq-')
        cls.log_path = os.path.join(cls.work, 'server.log')
        env = dict(os.environ, ENUNU_SERVER_PORT=PORT, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        cls.log = open(cls.log_path, 'w', encoding='utf-8')
        cls.server = subprocess.Popen([_common.PYTHON_EXE, 'enunu_server.py'], cwd=_common.SERVER_DIR,
                                      env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        cls.ctx = zmq.Context()
        try:
            deadline = time.time() + 120
            while time.time() < deadline:
                with open(cls.log_path, encoding='utf-8', errors='ignore') as f:
                    if 'Started enunu server' in f.read():
                        break
                if cls.server.poll() is not None:
                    raise RuntimeError('server exited during startup (port in use?)')
                time.sleep(0.5)
            else:
                raise RuntimeError('server did not start')
        except Exception:
            # setUpClass が失敗すると tearDownClass は呼ばれないので、ここでサーバーを止める
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(30)
        cls.log.close()
        cls.ctx.term()
        shutil.rmtree(cls.work, ignore_errors=True)

    def send(self, request, timeout=300):
        # OpenUtau の EnunuClient と同じく、リクエストごとに REQ ソケットを作る
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, timeout * 1000)
        sock.connect(f'tcp://localhost:{PORT}')
        try:
            sock.send_string(json.dumps(request))
            return json.loads(sock.recv())
        finally:
            sock.close()

    def server_log(self):
        with open(self.log_path, encoding='utf-8', errors='ignore') as f:
            return f.read()

    def test_1_requires_ver_check(self):
        # サーバーは ver_check を受けるまで他のコマンドを受け付けない (従来どおり)
        res = self.send(['acoustic', 'x.tmp', '', 'hash', '600'])
        self.assertEqual(res.get('error'), 'run ver_check.')

    def test_2_ver_check(self):
        res = self.send(['ver_check'])
        result = res['result']
        # 旧クライアントが見る項目は変えない
        self.assertEqual((result['name'], result['version'], result['author']),
                         ('SimpleENUNUServer', '1.0.0', 'roku10shi'))
        self.assertIn('config', result['features']['commands'])
        self.assertEqual(result['features']['diffusion']['mgc'], {'method': 'ddim', 'steps': 25})

    def test_3_config(self):
        self.send(['ver_check'])
        res = self.send(['config', {'diffusion': {'steps': 10}}])
        self.assertEqual(res['result']['diffusion']['mel'], {'method': 'ddim', 'steps': 10})
        res = self.send(['config', {'diffusion': {'mgc': 'foo:3'}}])
        self.assertIn('unknown diffusion method', res['error'])
        res = self.send(['config', 'not a dict'])
        self.assertIn('error', res)
        res = self.send(['config', {'diffusion': {'reset': True}}])
        self.assertEqual(res['result']['diffusion']['mel'], {'method': 'ddim', 'steps': 25})

    def test_4_unknown_command(self):
        self.send(['ver_check'])
        res = self.send(['nope', 'x.tmp', '', 'hash', '600'])
        self.assertIn('error', res)

    def test_5_openutau_flow_and_engine_expiry(self):
        voices = _common.test_voices()
        if not voices:
            self.skipTest('テスト用の音源が見つからない')
        voice = voices[0]
        tmp = _common.write_tmp(os.path.join(self.work, 'enu-z.tmp'), voice, _common.PHRASE_A)
        tmp2 = _common.write_tmp(os.path.join(self.work, 'enu-y.tmp'), voice, _common.PHRASE_B)
        self.send(['ver_check'])
        # EnunuRenderer と同じ 5 要素のリクエスト (duration=5 秒で有効期限を試す)
        res = self.send(['acoustic', tmp, '', 'X', '5'])
        self.assertNotIn('error', res)
        for key in ('path_f0', 'path_spectrogram', 'path_aperiodicity', 'path_mel', 'path_vuv'):
            self.assertIn(key, res['result'])
        self.assertTrue(os.path.isfile(res['result']['path_f0']))
        wav = os.path.join(self.work, 'z.wav')
        res = self.send(['synthe', tmp, wav, 'X', '5'])
        self.assertEqual(res['result']['path_wav'], wav)
        self.assertTrue(os.path.getsize(wav) > 1000)
        loads = self.server_log().count('Loading models')
        # 3 秒おきに使い続ける間は破棄されない
        for _ in range(3):
            time.sleep(3)
            self.send(['pitch', tmp, '', 'X', '5'])
        self.assertEqual(self.server_log().count('Loading models'), loads)
        # 別の歌手だけを使い続けると X は破棄され、次に使うと読み込み直す
        self.send(['pitch', tmp2, '', 'Y', '600'])
        time.sleep(6)
        self.send(['pitch', tmp2, '', 'Y', '600'])
        self.assertIn('release engine: X', self.server_log())
        self.send(['pitch', tmp, '', 'X', '5'])
        self.assertEqual(self.server_log().count('Loading models'), loads + 2)


    def test_6_synthe_after_tmp_deleted(self):
        voices = _common.test_voices()
        if not voices:
            self.skipTest('テスト用の音源が見つからない')
        tmp = _common.write_tmp(os.path.join(self.work, 'enu-w.tmp'), voices[-1], _common.PHRASE_A)
        self.send(['ver_check'])
        self.assertNotIn('error', self.send(['acoustic', tmp, '', 'W', '600']))
        os.remove(tmp)   # OpenUtau の「選択ノートのキャッシュ削除」相当
        # まだ読み込まれていない歌手 (再起動後相当) でも、ワークフォルダの temp.ust から読み込んで合成できる
        wav = os.path.join(self.work, 'w.wav')
        res = self.send(['synthe', tmp, wav, 'V', '600'])
        self.assertNotIn('error', res)
        self.assertTrue(os.path.getsize(wav) > 1000)


if __name__ == '__main__':
    unittest.main()
