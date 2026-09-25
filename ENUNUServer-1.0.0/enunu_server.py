#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import gc
import hashlib
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
    # name / version / author は旧クライアントが参照するので変えない。
    # 'features' は追加項目 (未対応クライアントは無視する)。
    return{
        'name': 'SimpleENUNUServer',
        'version': '1.0.0',
        'author': 'roku10shi',
        'features': {
            'commands': ['timing', 'acoustic', 'pitch', 'acoustic_f0', 'synthe', 'config'],
            'style_shift': True,
            'pitch_n_frames': True,
            'diffusion': enunu.diffusion_settings(),
        },
    }

def parse_style_shift(request, index):
    """request[index] を style_shift (半音, int) として読む。

    旧クライアントは送らないので、無い・数値でない場合は 0 にする。
    """
    if len(request) <= index:
        return 0
    value = request[index]
    if value is None or isinstance(value, (bool, list, dict)):
        return 0
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return 0

def ust_digest(engine: enunu.ENUNU):
    """このリクエストの UST (一時フォルダに複製した直後の内容) のハッシュ。

    run_timing が temp.ust を書き換えるので、各コマンドの先頭で呼ぶこと。
    """
    with open(engine.path_ust, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()

def features_meta(engine: enunu.ENUNU, kind, style_shift, digest, **extra):
    # 拡散設定が変わったら (環境設定でステップ数を変えた等) キャッシュは使わない
    return {'kind': kind, 'style_shift': style_shift, 'ust': digest,
            'feature_type': engine.feature_type, 'diffusion': enunu.diffusion_settings(), **extra}

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

def pitch_meta_path(engine: enunu.ENUNU):
    return os.path.splitext(engine.path_pitch_npy)[0] + '.json'

def pitch_cache_valid(engine: enunu.ENUNU, meta):
    try:
        with open(pitch_meta_path(engine), encoding='utf-8') as f:
            return json.load(f) == meta
    except (OSError, ValueError):
        return False

def cached_pitch_lf0(engine: enunu.ENUNU, digest, style_shift):
    """同じ UST・style_shift の pitch で保存した lf0_model の出力があれば返す (acoustic で lf0_model を省く)。"""
    if not engine.supports_lf0_conditioning() or not os.path.isfile(engine.path_pitch_lf0_npy):
        return None
    if not pitch_cache_valid(engine, features_meta(engine, 'pitch', style_shift, digest)):
        return None
    print('reuse lf0 from pitch')
    return np.load(engine.path_pitch_lf0_npy)

def array_digest(array):
    return hashlib.sha1(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()

def save_features(engine: enunu.ENUNU, kind, style_shift, digest, **extra):
    """音響特徴量をそのまま features.npz に保存する (ボコーダ合成・ピッチ差し替え用のキャッシュ)。"""
    meta = features_meta(engine, kind, style_shift, digest, **extra)
    arrays = {f's{i}': np.asarray(a) for i, a in enumerate(engine.multistream_features)}
    np.savez(engine.path_features_npz, meta=np.array(json.dumps(meta)), **arrays)

def load_features(engine: enunu.ENUNU):
    """features.npz を読む。無い・壊れている・形式が違う場合は (None, None)。"""
    path = engine.path_features_npz
    if not path or not os.path.isfile(path):
        return None, None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            n = 4 if meta.get('feature_type') == 'world' else 3
            features = tuple(data[f's{i}'] for i in range(n))
    except Exception as e:
        print(f'load_features: ignore broken cache ({e})')
        return None, None
    if meta.get('feature_type') != engine.feature_type:
        return None, None
    return features, meta

def load_legacy_melf0(engine: enunu.ENUNU):
    """features.npz が無い旧バージョンのワークフォルダ用。mel.npy / vuv.npy / f0.npy から復元する。"""
    if engine.feature_type != 'melf0':
        return None
    paths = (engine.path_mel_npy, engine.path_vuv_npy, engine.path_f0_npy)
    if not all(p and os.path.isfile(p) for p in paths):
        return None
    mel = np.load(engine.path_mel_npy)
    vuv = np.load(engine.path_vuv_npy).reshape(-1, 1)
    lf0 = continuous_lf0(np.load(engine.path_f0_npy))
    return (mel, lf0, vuv)

def continuous_lf0(f0):
    """f0 [Hz] (無声=0) を、無声区間を線形補間した連続 lf0 (T, 1) にする。"""
    f0 = np.asarray(f0, dtype=np.float64).flatten()
    voiced = f0 > 0
    lf0 = np.zeros_like(f0)
    if voiced.any():
        idx = np.arange(len(f0))
        lf0 = np.interp(idx, idx[voiced], np.log(f0[voiced]))
    return lf0.reshape(-1, 1)

def apply_editor_f0(engine: enunu.ENUNU, features):
    """editorf0.npy があれば、キャッシュ済みの特徴量の lf0 をエディタのピッチに差し替える。

    旧クライアント互換: ピッチだけ変えたときはこのファイルを置いて synthe を呼ぶ。
    0 のフレームはモデルのピッチのまま。フレーム数が違う場合は重なる範囲だけ差し替える。
    """
    path = engine.path_editorf0_npy
    if not path or not os.path.isfile(path):
        return features
    editor_f0 = np.asarray(np.load(path), dtype=np.float64).flatten()
    features = list(features)
    lf0 = np.array(features[1], dtype=np.float64).reshape(-1, 1)
    if len(editor_f0) != len(lf0):
        print(f'synthe: editorf0 frames {len(editor_f0)} != features frames {len(lf0)}')
    n = min(len(editor_f0), len(lf0))
    voiced = editor_f0[:n] > 0
    lf0[:n][voiced, 0] = np.log(editor_f0[:n][voiced])
    features[1] = lf0.astype(np.asarray(features[1]).dtype)
    print('synthe: pitch replaced by editorf0.npy')
    return tuple(features)

def timing(engine: enunu.ENUNU):
    print('timing: start')
    enunu.run_timing(engine=engine,)
    
    for path in (engine.path_full_timing, engine.path_mono_timing):
        if path is None or not os.path.isfile(path):
            raise Exception(f'{datetime.now()} :`{os.path.basename(path) if path else "None"}` does not exist.')
    print('timing: end')
    return {
        'path_full_timing': engine.path_full_timing,
        'path_mono_timing': engine.path_mono_timing,
    }


def acoustic(engine: enunu.ENUNU, style_shift=0):
    print('acoustic: start')
    digest = ust_digest(engine)
    features, meta = load_features(engine)
    cached = (
        features is not None
        and meta == features_meta(engine, 'acoustic', style_shift, digest)
        and os.path.isfile(engine.path_f0_npy)
    )
    if cached:
        print('acoustic: use cached features')
        engine.multistream_features = features
    else:
        run_acoustic_pipeline(engine, style_shift, digest)
    print('acoustic: end')
    return {
        'path_f0': engine.path_f0_npy,
        'path_spectrogram': engine.path_spectrogram_npy,
        'path_aperiodicity': engine.path_aperiodicity_npy,
        'path_mel': engine.path_mel_npy,
        'path_vuv': engine.path_vuv_npy,
    }

def pitch(engine: enunu.ENUNU, style_shift=0):
    """ピッチ (F0) だけを推定する。

    lf0_model を持つモデルは声色を計算しないので acoustic より大幅に速い。
    結果は pitch_f0.npy (f0.npy とは別) に保存する。
    """
    print('pitch: start')
    digest = ust_digest(engine)
    # pitch_f0.npy の横に、作った条件を pitch_f0.json として残す
    meta = features_meta(engine, 'pitch', style_shift, digest)
    if os.path.isfile(engine.path_pitch_npy) and pitch_cache_valid(engine, meta):
        print('pitch: use cached pitch_f0.npy')
        f0 = np.load(engine.path_pitch_npy)
    else:
        enunu.run_timing(engine=engine,step='acoustic')
        seed_rng(engine, digest, style_shift)
        engine.last_lf0_raw = None
        f0 = enunu.run_pitch(engine=engine,style_shift=style_shift)
        # lf0_model の生の出力も残し、同じ UST の acoustic / acoustic_f0 で使い回す
        if engine.last_lf0_raw is not None:
            np.save(engine.path_pitch_lf0_npy, engine.last_lf0_raw)
        elif os.path.isfile(engine.path_pitch_lf0_npy):
            os.remove(engine.path_pitch_lf0_npy)
        with open(pitch_meta_path(engine), 'w', encoding='utf-8') as f:
            json.dump(meta, f)
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
    digest = ust_digest(engine)
    f0_digest = array_digest(editor_f0)
    features, meta = load_features(engine)
    if (features is not None
            and meta == features_meta(engine, 'acoustic_f0', style_shift, digest, editor_f0=f0_digest)
            and os.path.isfile(engine.path_f0_npy)):
        # 同じ UST・同じエディタのピッチで作った結果が残っている (再要求など)
        print('acoustic_f0: use cached features')
        engine.multistream_features = features
    else:
        enunu.run_timing(engine=engine,step='acoustic')
        seed_rng(engine, digest, style_shift)
        enunu.run_acoustic(engine=engine,editor_f0=editor_f0,style_shift=style_shift,
                           lf0_base=cached_pitch_lf0(engine, digest, style_shift))
        enunu.run_npy(engine=engine)
        save_features(engine, 'acoustic_f0', style_shift, digest, editor_f0=f0_digest)
    print('acoustic_f0: end')
    return {
        'path_f0': engine.path_f0_npy,
        'path_spectrogram': engine.path_spectrogram_npy,
        'path_aperiodicity': engine.path_aperiodicity_npy,
        'path_mel': engine.path_mel_npy,
        'path_vuv': engine.path_vuv_npy,
        'lf0_conditioning': engine.supports_lf0_conditioning(),
    }

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
    features, meta = load_features(engine)
    if cache_only:
        if features is None:
            features = load_legacy_melf0(engine)
        if features is None:
            raise FileNotFoundError('the UST file is missing and no cached features to synthesize from')
        # シードは通常の合成と同じく、キャッシュを作ったときの UST から決める
        digest = meta['ust'] if meta else ust_digest(engine)
    else:
        digest = ust_digest(engine)
        if features is None:
            features = load_legacy_melf0(engine)
        elif meta.get('ust') != digest:
            features = None
    if features is None:
        print('synthe: no cached features, running acoustic')
        run_acoustic_pipeline(engine, style_shift, digest)
        features = engine.multistream_features
    engine.multistream_features = apply_editor_f0(engine, features)
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

def run_acoustic_pipeline(engine: enunu.ENUNU, style_shift, digest):
    enunu.run_timing(engine=engine,step='acoustic')
    seed_rng(engine, digest, style_shift)
    enunu.run_acoustic(engine=engine,style_shift=style_shift,
                       lf0_base=cached_pitch_lf0(engine, digest, style_shift))
    enunu.run_npy(engine=engine)
    save_features(engine, 'acoustic', style_shift, digest)

def config(request_body, engine_dict):
    """サーバー全体の設定を変える (OpenUtau の環境設定から送る想定)。

    ["config", {"diffusion": {"steps": 25}}] のように request[1] に dict を置く。
    読み込み済みの全エンジンと、以後に読み込むエンジンに反映する。
    """
    if not isinstance(request_body, dict):
        raise ValueError('config expects a dict at request[1]')
    result = {}
    if 'diffusion' in request_body:
        settings = enunu.set_diffusion_settings(request_body['diffusion'])
        for engine, _, _ in engine_dict.values():
            engine.apply_diffusion_settings(settings)
        result['diffusion'] = settings
    return result

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


    support = False
    engine_dict = {}

    for message in poll_socket(socket):
        """
        request body
        request[0]:step,
        request[1]:ust_path,
        request[2]:wav_path,
        request[3]:singer_name,
        request[4]:duration,
        request[5]:style_shift (任意。acoustic_f0 では request[5] が f0 配列、request[6] が style_shift)
        """
        request = json.loads(message)
        print('Received request: %s' % request)

        response = {}
        engine,duration,request_time = None,600,None
        try:
            if request[0] == 'ver_check':
                support = True
                response['result'] = check()
            elif support and request[0] == 'config':
                response['result'] = config(request[1] if len(request) > 1 else None, engine_dict)
            elif support:
                path_plugin, cache_only = resolve_plugin_path(request)
                if request[3] in engine_dict:
                    engine,duration,request_time = engine_dict[request[3]]
                    enunu.update_path(path_plugin,engine)
                    # 使うたびに期限を延ばす (最後に使ってから duration 秒で破棄)
                    request_time = time.time()
                    engine_dict[request[3]] = engine,duration,request_time
                else:
                    duration = int(request[4])
                    engine = enunu.setup(path_plugin)
                    request_time = time.time()
                    engine_dict[request[3]] = engine,duration,request_time


                if request[0] == 'timing':
                    response['result'] = timing(engine)
                elif request[0] == 'acoustic':
                    response['result'] = acoustic(engine, parse_style_shift(request, 5))
                elif request[0] == 'pitch':
                    response['result'] = pitch(engine, parse_style_shift(request, 5))
                elif request[0] == 'acoustic_f0':
                    editor_f0 = np.asarray(request[5], dtype=np.float64)
                    response['result'] = acoustic_f0(engine, editor_f0, parse_style_shift(request, 6))
                elif request[0] == 'synthe':
                    response['result'] = synthe(request[2],engine, parse_style_shift(request, 5), cache_only)
                else:
                    raise NotImplementedError('unexpected command %s' % request[0])
            else:
                response['error'] = 'run ver_check.'
            
        except Exception as e:
            response['error'] = str(e)
            traceback.print_exc()
        finally:
            keys_to_delete = [
                key for key, (_, duration, timestamp) in engine_dict.items()
                    if time.time() - timestamp > duration
            ]
            for key in keys_to_delete:
                print(f'release engine: {key}')
                del engine_dict[key]
            if keys_to_delete:
                engine = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print('Sending response: %s' % response)
        socket.send_string(json.dumps(response))


if __name__ == '__main__':
    main()
