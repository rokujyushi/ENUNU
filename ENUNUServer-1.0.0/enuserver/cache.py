"""ワークフォルダ (<UST名>_enutemp) に置くキャッシュ。

- features.npz: 音響特徴量そのもの。作った条件 (meta) と一緒に保存し、同じ条件なら推論を省く。
- pitch_f0.npy / pitch_lf0.npy / pitch_f0.json: pitch コマンドの結果と、lf0_model の生の出力。
- editorf0.npy: OpenUtau が synthe の前に置くエディタのピッチ。
"""
import hashlib
import json
import os

import numpy as np

from enuserver import diffusion


def ust_digest(engine):
    """このリクエストの UST (一時フォルダに複製した直後の内容) のハッシュ。

    run_timing が temp.ust を書き換えるので、各コマンドの先頭で呼ぶこと。
    """
    with open(engine.path_ust, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def array_digest(array):
    return hashlib.sha1(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def features_meta(engine, kind, style_shift, digest, **extra):
    """キャッシュを作った条件。これが一致しなければキャッシュは使わない。"""
    # 拡散設定が変わったら (環境設定でステップ数を変えた等) キャッシュは使わない
    # postfilter: GV でパワーを保つようにする前のキャッシュ (ノイズが乗ることがある) を使わないため
    return {'kind': kind, 'style_shift': style_shift, 'ust': digest,
            'feature_type': engine.feature_type, 'diffusion': diffusion.diffusion_settings(),
            'postfilter': 'gv-energy', **extra}


# features.npz ---------------------------------------------------------------

def save_features(engine, kind, style_shift, digest, **extra):
    """音響特徴量をそのまま features.npz に保存する (ボコーダ合成・ピッチ差し替え用のキャッシュ)。"""
    meta = features_meta(engine, kind, style_shift, digest, **extra)
    arrays = {f's{i}': np.asarray(a) for i, a in enumerate(engine.multistream_features)}
    np.savez(engine.path_features_npz, meta=np.array(json.dumps(meta)), **arrays)


def load_features(engine):
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


def load_matching_features(engine, meta):
    """meta と同じ条件で作った features.npz があり、クライアントが読む npy も揃っていれば特徴量を返す。"""
    features, saved_meta = load_features(engine)
    if features is None or saved_meta != meta or not npy_outputs_exist(engine):
        return None
    return features


def npy_outputs_exist(engine):
    """クライアントが読む npy が揃っているか。sp/ap は OpenUtau が WORLD 合成するときだけ作る (svs_npy 参照)。"""
    paths = [engine.path_f0_npy]
    if engine.client_reads_world_params():
        paths += [engine.path_spectrogram_npy, engine.path_aperiodicity_npy]
    return all(p and os.path.isfile(p) for p in paths)


def load_legacy_melf0(engine):
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


# editorf0.npy ---------------------------------------------------------------

def apply_editor_f0(engine, features):
    """editorf0.npy があれば、キャッシュ済みの特徴量の lf0 をエディタのピッチに差し替える。

    OpenUtau は synthe の前にこのファイルを置く (lf0_conditioning: false のモデルはこの方法でエディタのピッチを使う)。
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


# pitch ----------------------------------------------------------------------

def pitch_meta_path(engine):
    """pitch_f0.npy の横に、作った条件を pitch_f0.json として残す。"""
    return os.path.splitext(engine.path_pitch_npy)[0] + '.json'


def pitch_cache_valid(engine, meta):
    try:
        with open(pitch_meta_path(engine), encoding='utf-8') as f:
            return json.load(f) == meta
    except (OSError, ValueError):
        return False


def save_pitch(engine, meta):
    """pitch_f0.npy (svs_pitch が保存済み) の条件と、lf0_model の生の出力を保存する。"""
    if engine.last_lf0_raw is not None:
        np.save(engine.path_pitch_lf0_npy, engine.last_lf0_raw)
    elif os.path.isfile(engine.path_pitch_lf0_npy):
        os.remove(engine.path_pitch_lf0_npy)
    with open(pitch_meta_path(engine), 'w', encoding='utf-8') as f:
        json.dump(meta, f)


def cached_pitch_lf0(engine, digest, style_shift):
    """同じ UST・style_shift の pitch で保存した lf0_model の出力があれば返す (acoustic で lf0_model を省く)。"""
    if not engine.supports_lf0_conditioning() or not os.path.isfile(engine.path_pitch_lf0_npy):
        return None
    if not pitch_cache_valid(engine, features_meta(engine, 'pitch', style_shift, digest)):
        return None
    print('reuse lf0 from pitch')
    return np.load(engine.path_pitch_lf0_npy)
