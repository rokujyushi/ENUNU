#! /usr/bin/env python3
# coding: utf-8
# fmt: off
print('Starting enunu server...')
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
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
    path_meta = os.path.splitext(engine.path_pitch_npy)[0] + '.json'
    meta = features_meta(engine, 'pitch', style_shift, digest)
    f0 = None
    if os.path.isfile(engine.path_pitch_npy) and os.path.isfile(path_meta):
        try:
            with open(path_meta, encoding='utf-8') as f:
                if json.load(f) == meta:
                    f0 = np.load(engine.path_pitch_npy)
                    print('pitch: use cached pitch_f0.npy')
        except Exception as e:
            print(f'pitch: ignore broken cache ({e})')
    if f0 is None:
        enunu.run_timing(engine=engine,step='acoustic')
        f0 = enunu.run_pitch(engine=engine,style_shift=style_shift)
        with open(path_meta, 'w', encoding='utf-8') as f:
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
        enunu.run_acoustic(engine=engine,editor_f0=editor_f0,style_shift=style_shift)
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

def synthe(out_wav_path: str,engine: enunu.ENUNU, style_shift=0):
    """ワークフォルダのキャッシュ (features.npz) から波形を合成する。

    キャッシュがあれば推論せず、editorf0.npy があればピッチだけ差し替える。
    style_shift はキャッシュが無く acoustic から作り直すときだけ使う。
    """
    print('synthe: start')
    digest = ust_digest(engine)
    features, meta = load_features(engine)
    if features is None:
        features = load_legacy_melf0(engine)
    elif meta.get('ust') != digest:
        features = None
    if features is None:
        print('synthe: no cached features, running acoustic')
        run_acoustic_pipeline(engine, style_shift, digest)
        features = engine.multistream_features
    engine.multistream_features = apply_editor_f0(engine, features)
    enunu.run_synthesizer(out_wav_path=out_wav_path,engine=engine)
    print('synthe: end')
    return {
        'path_wav': out_wav_path,
    }

def run_acoustic_pipeline(engine: enunu.ENUNU, style_shift, digest):
    enunu.run_timing(engine=engine,step='acoustic')
    enunu.run_acoustic(engine=engine,style_shift=style_shift)
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
    socket.bind('tcp://*:15556')
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
                if request[3] in engine_dict:
                    engine,duration,request_time = engine_dict[request[3]]
                    enunu.update_path(request[1],engine)
                    # 使うたびに期限を延ばす (最後に使ってから duration 秒で破棄)
                    request_time = time.time()
                    engine_dict[request[3]] = engine,duration,request_time
                else:
                    duration = int(request[4])
                    engine = enunu.setup(request[1])
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
                    response['result'] = synthe(request[2],engine, parse_style_shift(request, 5))
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
