#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import gc
import json
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
# ボコーダを決定的に動かすため (synthe 参照)。CUDA の初期化より前に設定する必要がある
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import enunu
from enuserver import cache, diffusion
try:
    import zmq
except ModuleNotFoundError:
    python_exe = os.path.join('.', 'python-3.12.10-embed-amd64', 'python.exe')
    command = [python_exe, '-m', 'pip', 'install', 'pyzmq']
    print('command:', command)
    subprocess.run(command, check=True)
    import zmq
# fmt: on

def check():
    # name / author は旧クライアントも参照するので変えない。
    # クライアントは version ではなく features.commands で使えるコマンドを判断する。
    return{
        'name': 'ENUNUServer',
        'version': '2.0.0',
        'author': 'roku10shi',
        'features': {
            'commands': ['timing', 'acoustic', 'pitch', 'acoustic_f0', 'synthe', 'config'],
            'style_shift': True,
            'pitch_n_frames': True,
            'diffusion': diffusion.diffusion_settings(),
        },
    }

def parse_style_shift(value):
    """request[5] を style_shift (半音, int) として読む。数値でない場合は 0 にする。"""
    if value is None or isinstance(value, (bool, list, dict)):
        return 0
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0

def seed_rng(engine: enunu.ENUNU, digest, style_shift=0):
    """UST のハッシュから乱数のシードを決める。

    lf0_model の dropout・拡散のノイズ・ボコーダのノイズが乱数なので、固定しないと
    同じフレーズでも合成し直すたびに音が微妙に変わる。
    lf0_model の後でも設定し直すので、lf0 を再利用してもしなくても結果は同じになる。
    """
    seed = int(digest[:8], 16) ^ ((style_shift & 0xff) << 24)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    engine.post_lf0_seed = seed + 1
    return seed

def npy_paths(engine: enunu.ENUNU):
    """acoustic / acoustic_f0 のレスポンス。ファイルはモデルの形式によって一部しか作らない。"""
    return {
        'path_f0': engine.path_f0_npy,
        'path_spectrogram': engine.path_spectrogram_npy,
        'path_aperiodicity': engine.path_aperiodicity_npy,
        'path_mel': engine.path_mel_npy,
        'path_vuv': engine.path_vuv_npy,
    }

def timing(engine: enunu.ENUNU):
    print('timing: start')
    enunu.run_timing(engine)

    for path in (engine.path_full_timing, engine.path_mono_timing):
        if path is None or not os.path.isfile(path):
            raise Exception(f'{datetime.now()} :`{os.path.basename(path) if path else "None"}` does not exist.')
    print('timing: end')
    return {
        'path_full_timing': engine.path_full_timing,
        'path_mono_timing': engine.path_mono_timing,
    }

def run_acoustic_pipeline(engine: enunu.ENUNU, style_shift, digest, editor_f0=None):
    """音響特徴量を推論して npy と features.npz に保存する。editor_f0 を渡すと acoustic_f0 として作る。"""
    if editor_f0 is None:
        kind, extra = 'acoustic', {}
    else:
        kind, extra = 'acoustic_f0', {'editor_f0': cache.array_digest(editor_f0)}
    enunu.run_score_as_timing(engine)
    seed_rng(engine, digest, style_shift)
    enunu.run_acoustic(engine=engine,editor_f0=editor_f0,style_shift=style_shift,
                       lf0_base=cache.cached_pitch_lf0(engine, digest, style_shift))
    enunu.run_npy(engine=engine)
    cache.save_features(engine, kind, style_shift, digest, **extra)

def acoustic(engine: enunu.ENUNU, style_shift=0):
    print('acoustic: start')
    digest = cache.ust_digest(engine)
    features = cache.load_matching_features(engine, cache.features_meta(engine, 'acoustic', style_shift, digest))
    if features is not None:
        print('acoustic: use cached features')
        engine.multistream_features = features
    else:
        run_acoustic_pipeline(engine, style_shift, digest)
    print('acoustic: end')
    return npy_paths(engine)

def pitch(engine: enunu.ENUNU, style_shift=0):
    """ピッチ (F0) だけを推定する。

    lf0_model を持つモデルは声色を計算しないので acoustic より大幅に速い。
    結果は pitch_f0.npy (f0.npy とは別) に保存する。
    """
    print('pitch: start')
    digest = cache.ust_digest(engine)
    meta = cache.features_meta(engine, 'pitch', style_shift, digest)
    if os.path.isfile(engine.path_pitch_npy) and cache.pitch_cache_valid(engine, meta):
        print('pitch: use cached pitch_f0.npy')
        f0 = np.load(engine.path_pitch_npy)
    else:
        enunu.run_score_as_timing(engine)
        seed_rng(engine, digest, style_shift)
        engine.last_lf0_raw = None
        f0 = enunu.run_pitch(engine=engine,style_shift=style_shift)
        # lf0_model の生の出力も残し、同じ UST の acoustic / acoustic_f0 で使い回す
        cache.save_pitch(engine, meta)
    print('pitch: end')
    return {
        'path_f0': engine.path_pitch_npy,
        'lf0_conditioning': engine.supports_lf0_conditioning(),
        'n_frames': int(len(f0)),
    }

def acoustic_f0(engine: enunu.ENUNU, editor_f0: np.ndarray, style_shift=0):
    """エディタのピッチを条件にして音響特徴量を作り直す。

    editor_f0: float64 配列 (Hz)。0 のフレームはモデル自身のピッチを使う。
    lf0_model を持たないモデルでは通常の acoustic と同じ結果になる。
    出力は acoustic と同じ (f0.npy は editor_f0 のピッチになる)。
    """
    print('acoustic_f0: start')
    digest = cache.ust_digest(engine)
    meta = cache.features_meta(engine, 'acoustic_f0', style_shift, digest,
                               editor_f0=cache.array_digest(editor_f0))
    features = cache.load_matching_features(engine, meta)
    if features is not None:
        # 同じ UST・同じエディタのピッチで作った結果が残っている (再要求など)
        print('acoustic_f0: use cached features')
        engine.multistream_features = features
    else:
        run_acoustic_pipeline(engine, style_shift, digest, editor_f0)
    print('acoustic_f0: end')
    return {**npy_paths(engine), 'lf0_conditioning': engine.supports_lf0_conditioning()}

def resolve_plugin_path(request):
    """リクエストの TMP のパスと、TMP が無いときにキャッシュだけで合成するかどうかを返す。

    OpenUtau の「選択ノートのキャッシュ削除」は Cache 直下の enu-*.tmp を消すが _enutemp は残すので、
    その後の synthe では TMP が無い。この場合はワークフォルダに残っている temp.ust を代わりに使い、
    features.npz から合成する (temp.ust は拡張機能で編集済みなので UST ハッシュは照合しない)。
    """
    path_plugin = request[1].strip('"\'')
    if request[0] != 'synthe' or os.path.isfile(path_plugin):
        return path_plugin, False
    saved_ust = os.path.join(os.path.splitext(path_plugin)[0] + '_enutemp', 'temp.ust')
    if not os.path.isfile(saved_ust):
        raise FileNotFoundError(f'{path_plugin} and its work folder do not exist')
    print(f'synthe: {os.path.basename(path_plugin)} is missing, using cached features in the work folder')
    return saved_ust, True

def synthe(out_wav_path: str,engine: enunu.ENUNU, style_shift=0, cache_only=False):
    """ワークフォルダのキャッシュ (features.npz) から波形を合成する。

    キャッシュがあれば推論せず、editorf0.npy があればピッチだけ差し替える。
    style_shift はキャッシュが無く acoustic から作り直すときだけ使う。
    cache_only: TMP が無い場合 (resolve_plugin_path 参照)。UST を照合せずキャッシュを使い、無ければエラー。
    """
    print('synthe: start')
    features, meta = cache.load_features(engine)
    if cache_only:
        if features is None:
            features = cache.load_legacy_melf0(engine)
        if features is None:
            raise FileNotFoundError('the UST file is missing and no cached features to synthesize from')
        # シードは通常の合成と同じく、キャッシュを作ったときの UST から決める
        digest = meta['ust'] if meta else cache.ust_digest(engine)
    else:
        digest = cache.ust_digest(engine)
        if features is None:
            features = cache.load_legacy_melf0(engine)
        elif meta.get('ust') != digest:
            features = None
    if features is None:
        print('synthe: no cached features, running acoustic')
        run_acoustic_pipeline(engine, style_shift, digest)
        features = engine.multistream_features
    engine.multistream_features = cache.apply_editor_f0(engine, features)
    seed_rng(engine, digest, style_shift)
    # シードを固定しても GPU の一部の演算が非決定的で波形がわずかに変わるので、合成中だけ決定的な実装を使う
    # (ボコーダが 0.01〜0.03 秒ほど遅くなる。音響モデルはこれが無くても一致する)
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        enunu.run_synthesizer(out_wav_path=out_wav_path,engine=engine)
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)
    print('synthe: end')
    return {
        'path_wav': out_wav_path,
    }

def config(request_body, engines):
    """サーバー全体の設定を変える (OpenUtau は acoustic 系のリクエストの前に送る)。

    ["config", {"diffusion": {"steps": 25}}] のように request[1] に dict を置く。
    読み込み済みの全エンジン (engines) と、以後に読み込むエンジンに反映する。
    """
    if not isinstance(request_body, dict):
        raise ValueError('config expects a dict at request[1]')
    result = {}
    if 'diffusion' in request_body:
        settings = diffusion.set_diffusion_settings(request_body['diffusion'])
        for engine in engines:
            engine.apply_diffusion_settings(settings)
        result['diffusion'] = settings
    return result

class EnginePool:
    """歌手 (request[3]) ごとに読み込んだエンジン。最後に使ってから duration 秒で破棄する。"""

    def __init__(self):
        self._entries = {}  # key -> [engine, duration, 最後に使った時刻]

    def engines(self):
        return [engine for engine, _, _ in self._entries.values()]

    def get(self, key, path_plugin, duration):
        """key のエンジンを path_plugin のワークフォルダに切り替えて返す。無ければ読み込む。

        duration は読み込むときだけ使う。
        """
        entry = self._entries.get(key)
        if entry is not None:
            enunu.update_path(path_plugin, entry[0])
        else:
            duration = int(duration)
            entry = [enunu.setup(path_plugin), duration, None]
            self._entries[key] = entry
        # 使うたびに期限を延ばす
        entry[2] = time.time()
        return entry[0]

    def release_expired(self):
        now = time.time()
        expired = [key for key, (_, duration, last_used) in self._entries.items()
                   if now - last_used > duration]
        for key in expired:
            print(f'release engine: {key}')
            del self._entries[key]
        if expired:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def handle(request, pool: EnginePool):
    """リクエストを処理してレスポンスの result を返す。

    request (JSON 配列):
      ["ver_check"]
      ["config", {...}]
      [command, ust_path, wav_path, singer_name(hash), duration, style_shift]
      ["acoustic_f0", ust_path, wav_path, singer_name(hash), duration, style_shift, editor_f0]
    ver_check より前のコマンドも受け付ける (以前は 'run ver_check.' を返していたので、
    サーバーを再起動するとクライアントが ver_check を送り直すまで合成できなかった)。
    """
    command = request[0]
    if command == 'ver_check':
        return check()
    if command == 'config':
        return config(request[1] if len(request) > 1 else None, pool.engines())

    path_plugin, cache_only = resolve_plugin_path(request)
    engine = pool.get(request[3], path_plugin, request[4])
    style_shift = parse_style_shift(request[5])
    if command == 'timing':
        return timing(engine)
    if command == 'acoustic':
        return acoustic(engine, style_shift)
    if command == 'pitch':
        return pitch(engine, style_shift)
    if command == 'acoustic_f0':
        return acoustic_f0(engine, np.asarray(request[6], dtype=np.float64), style_shift)
    if command == 'synthe':
        return synthe(request[2], engine, style_shift, cache_only)
    raise NotImplementedError('unexpected command %s' % command)

def poll_socket(socket, timetick = 100):
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    # wait up to 100msec
    try:
        while True:
            obj = dict(poller.poll(timetick))
            if socket in obj and obj[socket] == zmq.POLLIN:
                yield socket.recv()
    except KeyboardInterrupt:
        pass
    # Escape while loop if there's a keyboard interrupt.


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    # OpenUtau は 15556 に接続する。ENUNU_SERVER_PORT はテストなどで別ポートを使うため
    socket.bind(f"tcp://*:{os.environ.get('ENUNU_SERVER_PORT', '15556')}")
    print('Started enunu server')

    pool = EnginePool()
    for message in poll_socket(socket):
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        try:
            response['result'] = handle(request, pool)
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()
        finally:
            pool.release_expired()

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))


if __name__ == '__main__':
    main()
