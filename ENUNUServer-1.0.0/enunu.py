#!/usr/bin/env python3
# Copyright (c) 2021-2025 oatsu
"""
1. UTAUプラグインのテキストファイルを読み取る。
2. LABファイル→WAVファイル
"""

import colored_traceback.auto  # noqa: F401

from contextlib import contextmanager
from importlib.util import find_spec
import logging
import shutil
import sys
import time
import tkinter
from argparse import ArgumentParser
from datetime import datetime
from glob import glob
import copy
import os
from os import chdir, listdir, makedirs, rename, startfile, remove
from os.path import (
    abspath,
    basename,
    dirname,
    exists,
    expanduser,
    join,
    relpath,
    splitext,
)
from tqdm.contrib.logging import logging_redirect_tqdm
from shutil import move
from tempfile import TemporaryDirectory, mkdtemp
from tkinter.filedialog import asksaveasfilename
from collections.abc import Iterable
import numpy as np
import utaupy
import yaml
from nnmnkwii.io import hts
from scipy.io import wavfile
from tqdm.auto import tqdm

# スクリプトのディレクトリをsys.pathに追加
sys.path.append(dirname(__file__))
import enulib

# scikit-learn で警告が出るのを無視
# import warnings
# warnings.simplefilter("ignore")

# my_package.my_moduleのみに絞ってsys.stderrにlogを出す
logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger('enunu')


SEGMENTED_SYNTHESIS = True

# torch をimportする。インストールされていない場合は新規インストールする ------
if find_spec('torch') is None:
    print('----------------------------------------------------------')
    print('初回起動ですね。')
    print('PC環境に合わせてPyTorchを自動インストールします。')
    print('インストール完了までしばらくお待ちください。')
    print('----------------------------------------------------------')
    enulib.install_torch.ltt_install_torch(sys.executable)
    print('----------------------------------------------------------')
    print('インストール成功しました。')
    print('----------------------------------------------------------\n')
import torch  # noqa: E402


# nnsvs 関連を import する ---------------------------------------------------
import nnsvs  # noqa: E402
from nnsvs.svs import SPSVS  # noqa: E402
# ↓EnunuServerCustom
from nnsvs.gen import gen_world_params
import pyworld
from nnmnkwii.frontend import merlin as fe
from nnmnkwii.preprocessing.f0 import interp1d
from nnsvs.base import PredictionType
from nnsvs.gen import _midi_to_hz
from nnsvs.pitch import lowpass_filter
from sklearn.preprocessing import MinMaxScaler
# ↑EnunuServerCustom
from enulib import enunu2nnsvs  # noqa: E402


def get_project_path(path_utauplugin):
    """
    キャッシュパスとプロジェクトパスを取得する。
    """
    plugin = utaupy.utauplugin.load(path_utauplugin)
    setting = plugin.setting
    # ustのパス
    path_ust = setting.get('Project')
    # 音源フォルダ
    voice_dir = setting['VoiceDir']
    # 音声キャッシュのフォルダ(LABとJSONを設置する)
    cache_dir = setting['CacheDir']

    return path_ust, voice_dir, cache_dir


def estimate_bit_depth(wav: np.ndarray) -> str:
    """
    wavformのビット深度を判定する。
    16bitか32bit
    16bitの最大値: 32767
    32bitの最大値: 2147483647
    """
    # 音量の最大値を取得
    max_gain = np.nanmax(np.abs(wav))
    # 学習データのビット深度を推定(8388608=2^24)
    if max_gain > 8388608:
        return 'int32'
    if max_gain > 8:
        return 'int16'
    return 'float'


def wrapped_enunu2nnsvs(voice_dir, out_dir):
    """ENUNU用のディレクトリ構造のモデルをNNSVS用に再構築する。"""
    # torch.save() の出力パスに日本語が含まれているとセーブできないので、一時フォルダを作ってそこに保存してから移動する。
    with TemporaryDirectory(prefix='.temp-enunu2nnsvs-', dir='.') as temp_dir:
        enunu2nnsvs.main(voice_dir, relpath(temp_dir))
        for path in listdir(temp_dir):
            move(join(temp_dir, path), join(out_dir, path))
    with open(join(voice_dir, 'enuconfig.yaml'), encoding='utf-8') as f:
        enuconfig = yaml.safe_load(f)
    rename(
        join(out_dir, 'kana2phonemes.table'),
        join(out_dir, basename(enuconfig['table_path'])),
    )


def packed_model_exists(voice_dir: str) -> bool:
    """フォルダ内にNNSVSモデルがあるかどうかを返す

    Args:
        dir (str): Path of the directory
    """
    # SPSVSクラスを使う際に必要なNNSVSモデル用のファイル(の一部)
    required_files = {'config.yaml', 'qst.hed'}
    # 全ての要求ファイルがフォルダ内に存在するか調べて返す
    return all(map(exists, [join(voice_dir, p) for p in required_files]))


def find_table(model_dir: str) -> str:
    """歌詞→音素の変換テーブルを探す"""
    table_files = glob(join(model_dir, '*.table'))
    if len(table_files) == 0:
        raise FileNotFoundError(f'Table file does not exist in {model_dir}.')
    if len(table_files) > 1:
        logger.warning('Multiple table files are found. : %s', table_files)
    logger.info('Using %s', basename(table_files[0]))
    return table_files[0]


def adjust_wav_gain_for_float32(wav: np.ndarray):
    """
    wavformのビット深度を判定して、float32で適切な音量で出力する。
    16bitか32bit
    16bitの最大値: 32767
    32bitの最大値: 2147483647
    ビット深度を指定してファイル出力(32bit float)

    """
    # 音量の最大値を取得
    max_gain = np.nanmax(np.abs(wav))

    # 学習データのビット深度を推定(8388608=2^24)
    # int32 -> float
    if max_gain > 8388608:
        return wav / 2147483647
    # int16 -> float
    if max_gain > 8:
        return wav / 32767
    # float
    return wav


class ENUNU(SPSVS):
    """ENUNU で合成するするときのクラス。

    Args:
        model_dir (str): NNSVSのモデルがあるフォルダ
        device (str): 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model_dir: str,
        device=None,
        verbose=0,
        **kwargs,
    ):
        # automatic device select
        if device is None:
            device = (
                torch.accelerator.current_accelerator()
                if torch.accelerator.is_available()
                else torch.device('cpu')
            )
        # initialize
        super().__init__(model_dir, device=device, verbose=verbose, **kwargs)
        # self.voice_dir = None
        # self.path_plugin = None
        self.path_ust = None
        self.path_table = None
        self.path_full_score = None
        self.path_mono_score = None
        self.path_full_timing = None
        self.path_mono_timing = None
        self.path_mgc = None
        self.path_f0 = None
        self.path_vuv = None
        self.path_bap = None
        self.path_feedback = None
        # self.path_wav = None
# ↓EnunuServerCustom
        self.multistream_features = None
        self.path_f0_npy = None
        self.path_pitch_npy = None
        self.path_mel_npy = None
        self.path_vuv_npy = None
        self.path_spectrogram_npy = None
        self.path_aperiodicity_npy = None
        self.path_features_npz = None
        self.path_editorf0_npy = None
        self.path_question = None
        self.start_time = None
        # 拡散モデルのサンプリングを間引いて高速化する (設定は diffusion_settings() を参照)
        self.apply_diffusion_settings(diffusion_settings())
# ↑EnunuServerCustom

# ↓EnunuServerCustom
    def set_paths(self, temp_dir, path_feedback=None):
        """ファイル入出力のPATHを設定する"""
        # f'{songname}_temp.ust' →　'temp.ust' に変更
        self.path_ust = join(temp_dir, 'temp.ust')
        # tableは固定
        # self.path_table = join(temp_dir, 'temp.table')
        self.path_full_score = join(temp_dir, 'score.full')
        self.path_mono_score = join(temp_dir, 'score.lab')
        self.path_full_timing = join(temp_dir, 'timing.full')
        self.path_mono_timing = join(temp_dir, 'timing.lab')
        self.path_mgc = join(temp_dir, 'acoustic_mgc.csv')
        self.path_f0 = join(temp_dir, 'acoustic_f0.csv')
        self.path_vuv = join(temp_dir, 'acoustic_vuv.csv')
        self.path_bap = join(temp_dir, 'acoustic_bap.csv')
# ↑EnunuServerCustom
        if path_feedback is not None:
            self.path_feedback = path_feedback
# ↓EnunuServerCustom
        self.path_f0_npy = join(temp_dir, 'f0.npy')
        # pitch コマンドの出力。f0.npy と分けて acoustic のキャッシュ判定に影響させない
        self.path_pitch_npy = join(temp_dir, 'pitch_f0.npy')
        self.path_mel_npy = join(temp_dir, 'mel.npy')
        self.path_vuv_npy = join(temp_dir, 'vuv.npy')
        self.path_spectrogram_npy = join(temp_dir, 'spectrogram.npy')
        self.path_aperiodicity_npy = join(temp_dir, 'aperiodicity.npy')
        # 音響特徴量そのもの (ボコーダ合成用キャッシュ) と、旧クライアントが置くエディタのピッチ
        self.path_features_npz = join(temp_dir, 'features.npz')
        self.path_editorf0_npy = join(temp_dir, 'editorf0.npy')
        self.path_question = join(temp_dir, 'temp.hed')
# ↑EnunuServerCustom

    def get_extension_path_list(self, key) -> list[str]:
        """
        拡張機能のパスのリストを取得する。
        パスが複数指定されていてもひとつしか指定されていなくてもループできるように、リストを返す。
        """
        config = self.config
        # 拡張機能の項目がなければNoneを返す。
        if 'extensions' not in config:
            return []
        if config.extensions is None:
            return []
        # 目的の拡張機能のパスがあれば取得する。
        extension_list = config.extensions.get(key)
        if extension_list is None:
            return []
        if extension_list == '':
            return []
        if isinstance(extension_list, str):
            return [extension_list]
        if isinstance(extension_list, Iterable):
            return list(extension_list)
        # 空文字列でもNULLでもリストでも文字列でもない場合
        raise TypeError(
            'Extension path must be null or strings or list, '
            f'not {type(extension_list)} for {extension_list}'
        )

    def edit_ust(self, ust: utaupy.ust.Ust, key='ust_editor') -> utaupy.ust.Ust:
        """
        合成前に、外部ツールでUSTを編集する。
        複数ツール
        """
        # UST加工ツールのパスを取得
        extension_list = self.get_extension_path_list(key)
        # UST加工ツールが指定されていない時はSkip
        if len(extension_list) == 0:
            return ust

        # 念のためustファイルを最新データで上書きする
        ust.write(self.path_ust)
        # 外部ツールで ust を編集
        for path_extension in extension_list:
            logger.info('Editing UST with %s', path_extension)
            enulib.extensions.run_extension(
                path_extension,
                ust=self.path_ust,
                table=self.path_table,
                feedback=self.path_feedback,
            )
        # 編集後のustファイルを読み取る
        ust = utaupy.ust.load(self.path_ust)
        return ust

    def edit_score(self, score_labels, key='score_editor'):
        """
        USTから変換して生成したフルラベルを外部ツールで編集する。
        """
        # LAB加工ツールのパスを取得
        extension_list = self.get_extension_path_list(key)
        # LAB加工ツールが指定されていない時はSkip
        if len(extension_list) == 0:
            return score_labels
        # 外部ツールでラベルを編集
        for path_extension in extension_list:
            logger.info('Editing LAB (score) with %s', path_extension)
            enulib.extensions.run_extension(
                path_extension,
                ust=self.path_ust,
                table=self.path_table,
                feedback=self.path_feedback,
                full_score=self.path_full_score,
            )
        score_labels = hts.load(self.path_full_score).round_()
        return score_labels

    def edit_timing(self, duration_modified_labels, key='timing_editor'):
        """
        外部ツールでタイミング編集する
        """
        # タイミング加工ツールのパスを取得
        extension_list = self.get_extension_path_list(key)
        # 指定されていない場合はSkip
        if len(extension_list) == 0:
            return duration_modified_labels

        # 複数ツールのすべてについて処理実施する
        for path_extension in extension_list:
            tqdm.write(f'Editing timing with {path_extension}')
            # 変更前のモノラベルを読んでおく
            with open(self.path_mono_timing, encoding='utf-8') as f:
                str_mono_old = f.read()
            enulib.extensions.run_extension(
                path_extension,
                ust=self.path_ust,
                table=self.path_table,
                feedback=self.path_feedback,
                full_score=self.path_full_score,
                mono_score=self.path_mono_score,
                full_timing=self.path_full_timing,
                mono_timing=self.path_mono_timing,
            )
            # 変更後のモノラベルを読む
            with open(self.path_mono_timing, encoding='utf-8') as f:
                str_mono_new = f.read()
            # モノラベルの時刻が変わっていたらフルラベルに転写して、
            # そうでなければフルラベルの時刻をモノラベルに転写する。
            # NOTE: 歌詞は編集していないという前提で処理する。
            if enulib.extensions.str_has_been_changed(str_mono_old, str_mono_new):
                enulib.extensions.merge_mono_time_change_to_full(
                    self.path_mono_timing, self.path_full_timing
                )
            else:
                enulib.extensions.merge_full_time_change_to_mono(
                    self.path_full_timing, self.path_mono_timing
                )

        # 編集後のfull_timing を読み取る
        duration_modified_labels = hts.load(self.path_full_timing).round_()
        return duration_modified_labels

    def edit_acoustic(self, multistream_features, feature_type, key='acoustic_editor'):
        """
        外部ツールでピッチなどを編集する。
        """
        # Validate input tuple size matches feature_type
        if feature_type == 'world':
            assert len(multistream_features) == 4, (
                f'Expected 4-element tuple for world, got {len(multistream_features)}'
            )
        elif feature_type == 'melf0':
            assert len(multistream_features) == 3, (
                f'Expected 3-element tuple for melf0, got {len(multistream_features)}'
            )

        # acoustic加工ツールのパスを取得
        extension_list = self.get_extension_path_list(key)
        # ツールが指定されていない場合はSkip
        if len(extension_list) == 0:
            return multistream_features

        # 想定外のボコーダが指定された場合もSkip
        if feature_type not in ['world', 'melf0']:
            logger.warning(
                'Unknown feature_type "%s" is selected. Skipping acoustic editor.',
                feature_type,
            )
            return multistream_features

        # ツールが指定されている場合はCSV書き出し
        if feature_type == 'world':
            mgc, lf0, vuv, bap = multistream_features
            f0 = np.exp(lf0)
            np.savetxt(self.path_mgc, mgc, fmt='%.16f', delimiter=',')
            np.savetxt(self.path_f0, f0, fmt='%.16f', delimiter=',')
            np.savetxt(self.path_vuv, vuv, fmt='%.16f', delimiter=',')
            np.savetxt(self.path_bap, bap, fmt='%.16f', delimiter=',')
        elif feature_type == 'melf0':
            mgc, lf0, vuv = multistream_features
            f0 = np.exp(lf0)
            # CSV書き出し
            np.savetxt(self.path_mgc, mgc, fmt='%.16f', delimiter=',')
            np.savetxt(self.path_f0, f0, fmt='%.16f', delimiter=',')
            np.savetxt(self.path_vuv, vuv, fmt='%.16f', delimiter=',')

        # 複数ツールのすべてについて処理実施する
        for path_extension in extension_list:
            tqdm.write(f'Editing acoustic features with {path_extension}')
            enulib.extensions.run_extension(
                path_extension,
                ust=self.path_ust,
                table=self.path_table,
                feedback=self.path_feedback,
                full_score=self.path_full_score,
                mono_score=self.path_mono_score,
                full_timing=self.path_full_timing,
                mono_timing=self.path_mono_timing,
                mgc=self.path_mgc,
                f0=self.path_f0,
                vuv=self.path_vuv,
                bap=self.path_bap,
            )

        # 編集が終わったらCSV読み取り
        if feature_type == 'world':
            mgc = np.loadtxt(self.path_mgc, delimiter=',', dtype=np.float64)
            lf0 = np.log(np.loadtxt(self.path_f0, delimiter=',', dtype=np.float64)).reshape(-1, 1)
            vuv = np.loadtxt(self.path_vuv, delimiter=',', dtype=np.float64).reshape(-1, 1)
            bap = np.loadtxt(self.path_bap, delimiter=',', dtype=np.float64)
            # 統合
            multistream_features = (mgc, lf0, vuv, bap)
        elif feature_type == 'melf0':
            # 編集が終わったらCSV読み取り
            mgc = np.loadtxt(self.path_mgc, delimiter=',', dtype=np.float64)
            lf0 = np.log(np.loadtxt(self.path_f0, delimiter=',', dtype=np.float64)).reshape(-1, 1)
            vuv = np.loadtxt(self.path_vuv, delimiter=',', dtype=np.float64).reshape(-1, 1)
            # 統合
            multistream_features = (mgc, lf0, vuv)
        else:
            raise Exception('Unexpected Error')
        # ↓EnunuServerCustom
        for path in (self.path_mgc, self.path_f0, self.path_vuv,self.path_bap):
            if exists(path):
                remove(path)
        # ↑EnunuServerCustom
        return multistream_features

# ↓EnunuServerCustom

    def svs_timing(
        self,
        labels,
        vocoder_type='world',
        post_filter_type='gv',
        **kwargs
    ):
        """Synthesize waveform from HTS labels.
        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
        """
        self.start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")
        # Predict timinigs
        duration_modified_labels = self.predict_timing(labels)
        # NOTE: ここにタイミング補正のための割り込み処理を追加-----------
        # mono_score を出力
        if self.path_mono_score is not None:
            with open(self.path_mono_score, 'w', encoding='utf-8') as f:
                f.write(str(nnsvs.io.hts.full_to_mono(labels)))
        
        # mono_timing を出力
        if self.path_mono_timing is not None:
            with open(self.path_mono_timing, 'w', encoding='utf-8') as f:
                f.write(str(nnsvs.io.hts.full_to_mono(duration_modified_labels)))
        
        # full_timing を出力
        if self.path_full_timing is not None:
            with open(self.path_full_timing, 'w', encoding='utf-8') as f:
                f.write(str(duration_modified_labels))
    
    def svs_acoustic(
        self,
        vocoder_type='world',
        post_filter_type='gv',
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        style_shift=0,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        editor_f0=None,
        **kwargs
    ):
        """Synthesize waveform from HTS labels.
        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            style_shift (int): style shift parameter
            force_fix_vuv (bool): Whether to correct VUV.
            fill_silence_to_rest (bool): Fill silence to rest frames.
            editor_f0 (ndarray): エディタのピッチ [Hz] (フレーム単位, 0 は指定なし)。
                指定するとモデルのピッチの代わりにこれを条件にして声色などを生成する。
                lf0_model を持たないモデルでは無視される。
        """
        if(self.start_time == None):
            self.start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")
        # 編集後のfull_timing を読み取る
        duration_modified_labels = hts.load(self.path_full_timing).round_()
        # Run acoustic model and vocoder
        hts_frame_shift = int(self.config.frame_period * 1e4)
        duration_modified_labels.frame_shift = hts_frame_shift

        # Predict acoustic features
        # NOTE: if non-zero pre_f0_shift_in_cent is specified, the input pitch
        # will be shifted before running the acoustic model
        if editor_f0 is not None and self.supports_lf0_conditioning():
            # 後処理で -style_shift されるので、条件に使うピッチは先に +style_shift しておく
            editor_lf0 = np.full(len(editor_f0), np.nan)
            voiced = np.asarray(editor_f0) > 0
            editor_lf0[voiced] = np.log(np.asarray(editor_f0)[voiced]) \
                + style_shift * 100 * np.log(2) / 1200
            with self._override_lf0(editor_lf0):
                acoustic_features = self.predict_acoustic(
                    duration_modified_labels,
                    f0_shift_in_cent=style_shift * 100,
                )
        else:
            if editor_f0 is not None:
                logger.warning(
                    'This acoustic model has no separate lf0_model. '
                    'Editor pitch is ignored for acoustic feature prediction.'
                )
            acoustic_features = self.predict_acoustic(
                duration_modified_labels,
                f0_shift_in_cent=style_shift * 100,
            )

        # Post-processing for acoustic features
        # NOTE: if non-zero post_f0_shift_in_cent is specified, the output pitch
        # will be shifted as a part of post-processing
        self.multistream_features = self.postprocess_acoustic(
            acoustic_features=acoustic_features,
            duration_modified_labels=duration_modified_labels,
            trajectory_smoothing=trajectory_smoothing,
            trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
            trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
            force_fix_vuv=force_fix_vuv,
            fill_silence_to_rest=fill_silence_to_rest,
            f0_shift_in_cent=-style_shift * 100,
        )

        # NOTE: ここにピッチ補正のための割り込み処理を追加-----------
        self.multistream_features = self.edit_acoustic(
            self.multistream_features,
            feature_type=self.feature_type
        )

    def supports_lf0_conditioning(self) -> bool:
        """音響モデルがピッチ専用のサブモデル (lf0_model) を持ち、
        声色などをそのピッチを条件にして生成する構造かどうか。

        NNSVS の multistream 系 (NPSS* / *SeparateF0*) が該当する。
        旧来の一括予測モデル (Conv1dResnet, RMDN など) は False。
        """
        lf0_model = getattr(self.acoustic_model, 'lf0_model', None)
        return (
            lf0_model is not None
            and hasattr(self.acoustic_model, 'out_lf0_idx')
            and lf0_model.prediction_type() != PredictionType.PROBABILISTIC
        )

    def apply_diffusion_settings(self, settings: dict) -> None:
        """acoustic_model 配下の GaussianDiffusion ごとにサンプラとステップ数を設定する。

        settings: diffusion_settings() の戻り値 ({'mgc': {'method': 'ddim', 'steps': 25}, ...})。
        モジュール名に mgc / mel / bap を含むものにそれぞれの設定を、それ以外には 'other' を使う。
        何度呼んでもよい (config コマンドで実行中に変更する)。
        """
        try:
            from nnsvs.diffsinger.diffusion import GaussianDiffusion, extract
        except Exception as e:
            logger.warning("GaussianDiffusion を import できず拡散設定を適用できません: %s", e)
            return
        applied = []
        for name, m in self.acoustic_model.named_modules():
            if not isinstance(m, GaussianDiffusion):
                continue
            short = name.split(".")[-1] if name else "(root)"
            stream = next((k for k in DIFFUSION_STREAMS if k in short), 'other')
            method, steps = settings[stream]['method'], settings[stream]['steps']
            # 以前に差し替えたサンプラを外してから設定し直す
            m.__dict__.pop('p_sample_plms', None)
            interval = max(1, round(m.K_step / steps)) if method != 'ddpm' else 1
            if interval <= 1:
                m.pndm_speedup = None
                method = 'ddpm'
            else:
                m.pndm_speedup = interval
                if method == 'ddim':
                    _bind_ddim_sampler(m, extract)
                elif method == 'eta1':
                    _bind_eta1_sampler(m, extract)
            applied.append(f'{short}={method}:{-(-m.K_step // interval)}')
        if applied:
            logger.info("拡散サンプラ: %s", ", ".join(applied))

    def _acoustic_input(self, labels, f0_shift_in_cent=0):
        """音響モデルへの入力 (正規化済みの楽譜特徴量) と休符フレームのマスクを作る。

        nnsvs.gen.predict_acoustic の入力作成部分と同じ処理。
        """
        feats = fe.linguistic_features(
            labels,
            self.binary_dict,
            self.numeric_dict,
            add_frame_features=True,
            subphone_features=self.acoustic_config.get('subphone_features', 'coarse_coding'),
            frame_shift=int(self.config.frame_period * 1e4),
        )
        if self.config.log_f0_conditioning:
            for idx in self.pitch_indices:
                feats[:, idx] = interp1d(_midi_to_hz(feats, idx, True), kind='slinear')
                if f0_shift_in_cent != 0:
                    feats[:, idx] += f0_shift_in_cent * np.log(2) / 1200
        rest = feats[:, self.acoustic_model.in_rest_idx] > 0.5
        feats = self.acoustic_in_scaler.transform(feats)
        if self.acoustic_config.get('force_clip_input_features', True) \
                and isinstance(self.acoustic_in_scaler, MinMaxScaler):
            non_pitch = [i for i in range(feats.shape[1]) if i not in self.pitch_indices]
            feats[:, non_pitch] = np.clip(
                feats[:, non_pitch], *self.acoustic_in_scaler.feature_range
            )
        x = torch.from_numpy(feats).float().to(self.device).view(1, -1, feats.shape[1])
        return x, rest

    def _lf0_scale(self):
        """出力 lf0 の正規化パラメータ (mean, scale)"""
        idx = self.acoustic_model.out_lf0_idx
        return self.acoustic_out_scaler.mean_[idx], self.acoustic_out_scaler.scale_[idx]

    @torch.no_grad()
    def predict_lf0(self, labels, f0_shift_in_cent=0):
        """lf0_model だけを実行して連続 log-F0 を返す (声色などは計算しない)。

        Returns:
            (ndarray, ndarray): 連続 log-F0 (T,) と休符フレームのマスク (T,)
        """
        x, rest = self._acoustic_input(labels, f0_shift_in_cent)
        if hasattr(self.acoustic_model, '_set_lf0_params'):
            self.acoustic_model._set_lf0_params()
        lf0 = self.acoustic_model.lf0_model.inference(x, [x.shape[1]])
        lf0 = lf0.squeeze(0).cpu().numpy()[: len(rest), 0]
        mean, scale = self._lf0_scale()
        return lf0 * scale + mean, rest

    @contextmanager
    def _override_lf0(self, lf0_target):
        """lf0_model の出力を lf0_target (log-Hz, NaN のフレームはモデルの値のまま) に差し替える。

        multistream モデルは lf0_model の出力を条件にして mgc/bap/mel/vuv を生成するので、
        この間に predict_acoustic を呼ぶと、指定したピッチに合わせた特徴量が得られる。
        """
        lf0_model = self.acoustic_model.lf0_model
        original = lf0_model.inference
        mean, scale = self._lf0_scale()
        target = (np.asarray(lf0_target, dtype=np.float64) - mean) / scale

        def inference(x, lengths=None):
            pred = original(x, lengths)
            # 入力は reduction_factor の倍数にパディングされているので長さを合わせる
            t = np.full(pred.shape[1], np.nan)
            n = min(len(t), len(target))
            t[:n] = target[:n]
            mask = ~np.isnan(t)
            out = pred.clone()
            out[0, torch.from_numpy(mask).to(out.device), 0] = \
                torch.from_numpy(t[mask]).to(out)
            return out

        lf0_model.inference = inference
        try:
            yield
        finally:
            del lf0_model.inference

    def svs_pitch(
        self,
        style_shift=0,
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff_f0=20,
        **kwargs
    ):
        """ピッチ (F0 [Hz]) だけを推定して path_pitch_npy に保存する。

        lf0_model を持つモデルは lf0_model だけを実行する (拡散モデルなどを回さないので速い)。
        持たないモデルは音響モデル全体を実行して F0 だけを取り出す。
        休符 (無声) フレームは 0 になる。
        """
        start_time = time.time()
        labels = hts.load(self.path_full_timing).round_()
        labels.frame_shift = int(self.config.frame_period * 1e4)

        if self.supports_lf0_conditioning():
            lf0, rest = self.predict_lf0(labels, f0_shift_in_cent=style_shift * 100)
            lf0 -= style_shift * 100 * np.log(2) / 1200
            if trajectory_smoothing:
                modfs = int(1 / (self.config.frame_period * 0.001))
                lf0 = lowpass_filter(lf0, modfs, cutoff=trajectory_smoothing_cutoff_f0)
            f0 = np.exp(lf0)
            f0[rest] = 0
        else:
            self.svs_acoustic(style_shift=style_shift, force_fix_vuv=True, **kwargs)
            _, lf0, vuv = self.multistream_features[:3]
            lf0, vuv = lf0.flatten(), vuv.flatten()
            f0 = np.where(lf0 > 0, np.exp(lf0), 0)
            f0[vuv < 0.5] = 0

        np.save(self.path_pitch_npy, f0.astype(np.float64))
        logger.info(f'Elapsed time for pitch prediction: {time.time() - start_time:.3f} sec')
        return f0

    def svs_npy(
        self,
        vocoder_type='world',
        vuv_threshold=0.5,
        kind='linear',
        **kwargs
        ):
        """Synthesize waveform from HTS labels.
        Args:
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            vuv_threshold (float): Threshold for VUV.
            kind (str):
        """
        if(self.start_time == None):
            self.start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        
        data,f0, spectrogram, aperiodicity, mel, vuv = None,None,None,None,None,None
        if self.multistream_features is not None and len(self.multistream_features) >= 4:
            mgc, lf0, vuv, bap = self.multistream_features
            # Generate WORLD parameters
            f0, spectrogram, aperiodicity = gen_world_params(
                mgc, lf0, vuv, bap, self.config.sample_rate, vuv_threshold=vuv_threshold,use_world_codec=self.config.use_world_codec
            )
        elif self.multistream_features is not None and len(self.multistream_features) == 3:
            mel, lf0, vuv = self.multistream_features
            f0 = lf0.copy()
            f0[np.nonzero(f0)] = np.exp(f0[np.nonzero(f0)])
            f0[vuv < 0.5] = 0
            f0 = f0.flatten()
        else:
            data = self.predict_waveform(
                multistream_features=self.multistream_features,
                vocoder_type=vocoder_type,
                vuv_threshold=vuv_threshold,
            )
            data = data.astype(np.double)
            # Generate WAV to WORLD parameters
            f0, spectrogram, aperiodicity = pyworld.wav2world(data, self.config.sample_rate)
            f0 = self.interp1d(f0=f0,kind=kind)

        # npyとしてparameterの行列を出力
        for path, array, strtype in (
            (self.path_f0_npy, f0.astype(np.float64) if not f0 is None else None, 'f0'),
            (self.path_mel_npy, mel.astype(np.float64) if not mel is None else None, 'mel'),
            (self.path_vuv_npy, vuv.astype(np.float64) if not vuv is None else None, 'vuv'),
            (self.path_spectrogram_npy, spectrogram.astype(np.float64) if not spectrogram is None else None, 'spectrogram'),
            (self.path_aperiodicity_npy, aperiodicity.astype(np.float64) if not aperiodicity is None else None, 'aperiodicity'),
        ):
            if array is not None and path is not None:
                print(f'save {strtype}.npy')
                np.save(path, array)

        if(self.start_time != None):
            logger.info(f"Total time: {time.time() - self.start_time:.3f} sec")
            RT = (time.time() - self.start_time) / (len(f0) / self.config.sample_rate)
            logger.info(f"Total real-time factor: {RT:.3f}")


    
    def svs_synthe(
        self,
        vocoder_type='world',
        post_filter_type='gv',
        vuv_threshold=0.5,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
        **kwargs
    ):
        """Synthesize waveform from HTS labels.
        Args:
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            vuv_threshold (float): Threshold for VUV.
            dtype (np.dtype): Data type of the output waveform.
            peak_norm (bool): Whether to normalize the waveform by peak value.
            loudness_norm (bool): Whether to normalize the waveform by loudness.
            target_loudness (float): Target loudness in dB.
        """
        if(self.start_time == None):
            self.start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ['world', 'pwg', 'usfgan', 'auto']:
            raise ValueError(f'Unknown vocoder type: {vocoder_type}')
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")
        

        # Generate waveform by vocoder
        wav = self.predict_waveform(
            multistream_features=self.multistream_features,
            vocoder_type=vocoder_type,
            vuv_threshold=vuv_threshold,
        )
        # Post-processing for the output waveform
        wav = self.postprocess_waveform(
            wav,
            dtype=dtype,
            peak_norm=peak_norm,
            loudness_norm=loudness_norm,
            target_loudness=target_loudness,
        )


        logger.info(f"Total time: {time.time() - self.start_time:.3f} sec")
        RT = (time.time() - self.start_time) / (len(wav) / self.config.sample_rate)
        logger.info(f"Total real-time factor: {RT:.3f}")
        return wav, self.config.sample_rate

# ↑EnunuServerCustom

    def svs(
        self,
        labels,
        vocoder_type='world',
        post_filter_type='gv',
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        style_shift=0,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
        segmented_synthesis=False,
        **kwargs,
    ):
        """Synthesize waveform from HTS labels.
        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            style_shift (int): style shift parameter
            force_fix_vuv (bool): Whether to correct VUV.
            fill_silence_to_rest (bool): Fill silence to rest frames.
            dtype (np.dtype): Data type of the output waveform.
            peak_norm (bool): Whether to normalize the waveform by peak value.
            loudness_norm (bool): Whether to normalize the waveform by loudness.
            target_loudness (float): Target loudness in dB.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ['world', 'pwg', 'usfgan', 'auto']:
            raise ValueError(f'Unknown vocoder type: {vocoder_type}')
        if post_filter_type not in ['merlin', 'nnsvs', 'gv', 'none']:
            raise ValueError(f'Unknown post-filter type: {post_filter_type}')

        # Predict timinigs
        duration_modified_labels = self.predict_timing(labels)

        # NOTE: ここにタイミング補正のための割り込み処理を追加-----------
        # mono_score を出力
        with open(self.path_mono_score, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(labels)))
        # mono_timing を出力
        with open(self.path_mono_timing, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(duration_modified_labels)))
        # full_timing を出力
        with open(self.path_full_timing, 'w', encoding='utf-8') as f:
            f.write(str(duration_modified_labels))
        # 外部で加工した結果でタイミング情報を置換
        duration_modified_labels = self.edit_timing(duration_modified_labels)
        # ---------------------------------------------------------------

        # NOTE: segmented synthesis is not well tested. There MUST be better ways
        # to do this.
        if segmented_synthesis:
            # logger.warning('Segmented synthesis is not well tested. Use it on your own risk.')
            # NOTE: ここsegment_labels が nnsvs の中の関数にあるので呼び出せるように改造済み
            duration_modified_labels_segs = nnsvs.io.hts.segment_labels(
                duration_modified_labels,
                # the following parameters are based on experiments in the NNSVS's paper
                # tuned with Namine Ritsu's database
                silence_threshold=0.1,
                min_duration=5.0,
                force_split_threshold=5.0,
            )
        else:
            duration_modified_labels_segs = [duration_modified_labels]

        # Run acoustic model and vocoder
        hts_frame_shift = int(self.config.frame_period * 1e4)
        wavs = []
        logger.info('Number of segments: %s', len(duration_modified_labels_segs))
        with logging_redirect_tqdm(loggers=[self.logger]):
            for duration_modified_labels_seg in tqdm(
                duration_modified_labels_segs,
                colour='blue',
                desc='[segment]',
                total=len(duration_modified_labels_segs),
            ):
                duration_modified_labels_seg.frame_shift = hts_frame_shift

                # Predict acoustic features
                # NOTE: if non-zero pre_f0_shift_in_cent is specified, the input pitch
                # will be shifted before running the acoustic model
                acoustic_features = self.predict_acoustic(
                    duration_modified_labels_seg,
                    f0_shift_in_cent=style_shift * 100,
                )

                # Post-processing for acoustic features
                # NOTE: if non-zero post_f0_shift_in_cent is specified, the output pitch
                # will be shifted as a part of post-processing
                multistream_features = self.postprocess_acoustic(
                    acoustic_features=acoustic_features,
                    duration_modified_labels=duration_modified_labels_seg,
                    trajectory_smoothing=trajectory_smoothing,
                    trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
                    trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
                    force_fix_vuv=force_fix_vuv,
                    fill_silence_to_rest=fill_silence_to_rest,
                    f0_shift_in_cent=-style_shift * 100,
                )

                # NOTE: ここにピッチ補正のための割り込み処理を追加-----------
                multistream_features = self.edit_acoustic(
                    multistream_features, feature_type=self.feature_type
                )

                # Generate waveform by vocoder
                wav = self.predict_waveform(
                    multistream_features=multistream_features,
                    vocoder_type=vocoder_type,
                    vuv_threshold=vuv_threshold,
                )

                wavs.append(wav)

        # Concatenate segmented waveforms
        wav = np.concatenate(wavs, axis=0).reshape(-1)

        # Post-processing for the output waveform
        wav = self.postprocess_waveform(
            wav,
            dtype=dtype,
            peak_norm=peak_norm,
            loudness_norm=loudness_norm,
            target_loudness=target_loudness,
        )
        logger.info(f'Total time: {time.time() - start_time:.3f} sec')
        RT = (time.time() - start_time) / (len(wav) / self.sample_rate)
        logger.info(f'Total real-time factor: {RT:.3f}')
        return wav, self.sample_rate

# ↓EnunuServerCustom

DIFFUSION_STREAMS = ('mgc', 'mel', 'bap')
DIFFUSION_METHODS = ('ddpm', 'ddim', 'plms', 'eta1')
# 既定値。mgc / mel は DDIM 25 ステップで 100 ステップとほぼ同等の品質 (2026-09-25 検証)。
DEFAULT_DIFFUSION = {
    'mgc': {'method': 'ddim', 'steps': 25},
    'mel': {'method': 'ddim', 'steps': 25},
    'bap': {'method': 'plms', 'steps': 10},
    'other': {'method': 'ddpm', 'steps': 100},
}
# config コマンドで変更された設定 (None なら環境変数・既定値)
_diffusion_override = None


def parse_diffusion_spec(spec: str) -> dict:
    """ 'ddim:25' / 'ddpm' / '25' (手法は既定のまま) を {'method', 'steps'} にする。"""
    method, _, steps = spec.strip().lower().partition(':')
    if method.isdigit() and not steps:
        return {'steps': int(method)}
    if method not in DIFFUSION_METHODS:
        raise ValueError(f'unknown diffusion method: {method}')
    result = {'method': method}
    if steps:
        result['steps'] = int(steps)
    return result


def _legacy_env_diffusion() -> dict | None:
    """旧環境変数 (ENUNU_DIFFUSION_SPEEDUP / TARGETS / METHOD) が指定されていれば、その意味どおりの設定を返す。"""
    keys = ("ENUNU_DIFFUSION_SPEEDUP", "ENUNU_DIFFUSION_TARGETS", "ENUNU_DIFFUSION_METHOD")
    if not any(k in os.environ for k in keys):
        return None
    try:
        speedup = max(int(os.environ.get("ENUNU_DIFFUSION_SPEEDUP", "10")), 1)
    except ValueError:
        speedup = 10
    targets = os.environ.get("ENUNU_DIFFUSION_TARGETS", "bap").lower()
    method = os.environ.get("ENUNU_DIFFUSION_METHOD", "plms").lower()
    if method not in DIFFUSION_METHODS:
        method = 'plms'
    settings = {k: {'method': 'ddpm', 'steps': 100} for k in (*DIFFUSION_STREAMS, 'other')}
    if speedup > 1:
        for k in DIFFUSION_STREAMS:
            if targets == 'all' or targets in k:
                settings[k] = {'method': method, 'steps': max(1, 100 // speedup)}
    return settings


def diffusion_settings() -> dict:
    """拡散モデルのサンプラ設定を返す。優先順: config コマンド > 環境変数 > 既定値。

    環境変数:
      ENUNU_DIFFUSION_MGC / _MEL / _BAP : 'ddim:25' のように手法:ステップ数
      (旧) ENUNU_DIFFUSION_SPEEDUP / TARGETS / METHOD : 指定されている場合は従来の意味で解釈
    """
    if _diffusion_override is not None:
        return copy.deepcopy(_diffusion_override)
    settings = _legacy_env_diffusion() or copy.deepcopy(DEFAULT_DIFFUSION)
    for k in DIFFUSION_STREAMS:
        spec = os.environ.get(f"ENUNU_DIFFUSION_{k.upper()}")
        if spec:
            try:
                settings[k].update(parse_diffusion_spec(spec))
            except ValueError as e:
                logger.warning("ENUNU_DIFFUSION_%s を無視します: %s", k.upper(), e)
    return settings


def set_diffusion_settings(request: dict) -> dict:
    """config コマンドの diffusion 設定を検証して反映し、反映後の設定を返す。

    request 例: {'steps': 25}  (mgc と mel のステップ数だけ変える)
               {'mgc': {'method': 'ddim', 'steps': 25}, 'bap': 'plms:10'}
               {'reset': True}  (環境変数・既定値に戻す)
    """
    global _diffusion_override
    if request.get('reset'):
        _diffusion_override = None
        return diffusion_settings()
    settings = diffusion_settings()
    if 'steps' in request:
        for k in ('mgc', 'mel'):
            settings[k]['steps'] = request['steps']
    for k in DIFFUSION_STREAMS:
        if k in request:
            value = request[k]
            settings[k].update(parse_diffusion_spec(value) if isinstance(value, str) else value)
    for k, v in settings.items():
        if v.get('method') not in DIFFUSION_METHODS:
            raise ValueError(f'unknown diffusion method for {k}: {v.get("method")}')
        if not isinstance(v.get('steps'), int) or isinstance(v['steps'], bool) or v['steps'] < 1:
            raise ValueError(f'diffusion steps for {k} must be a positive integer')
    _diffusion_override = settings
    return copy.deepcopy(settings)


def _bind_ddim_sampler(gauss_diffusion, extract_fn) -> None:
    """GaussianDiffusion の p_sample_plms を DDIM (η=0) に置き換える。"""
    import types

    @torch.no_grad()
    def ddim_step(self, x, t, interval, cond):
        a_t = extract_fn(self.alphas_cumprod, t, x.shape)
        a_prev = extract_fn(
            self.alphas_cumprod,
            torch.max(t - interval, torch.zeros_like(t)),
            x.shape,
        )
        sqrt_a_t = a_t.sqrt()
        sqrt_one_minus_a_t = (1 - a_t).sqrt()
        sqrt_a_prev = a_prev.sqrt()
        sqrt_one_minus_a_prev = (1 - a_prev).sqrt()
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x0_hat = (x - sqrt_one_minus_a_t * noise_pred) / sqrt_a_t
        x0_hat = x0_hat.clamp(-1.0, 1.0)
        if bool((t == 0).all()):
            return x0_hat
        eps_eff = (x - sqrt_a_t * x0_hat) / sqrt_one_minus_a_t.clamp(min=1e-8)
        return sqrt_a_prev * x0_hat + sqrt_one_minus_a_prev * eps_eff

    gauss_diffusion.p_sample_plms = types.MethodType(ddim_step, gauss_diffusion)


def _bind_eta1_sampler(gauss_diffusion, extract_fn) -> None:
    """GaussianDiffusion の p_sample_plms を η=1 strided ancestral に置き換える。"""
    import types

    @torch.no_grad()
    def eta1_step(self, x, t, interval, cond):
        t_prev = torch.max(t - interval, torch.zeros_like(t))
        a_t = extract_fn(self.alphas_cumprod, t, x.shape)
        a_prev = extract_fn(self.alphas_cumprod, t_prev, x.shape)
        sqrt_a_t = a_t.sqrt()
        sqrt_one_minus_a_t = (1 - a_t).sqrt().clamp(min=1e-8)
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x0_hat = ((x - sqrt_one_minus_a_t * noise_pred) / sqrt_a_t).clamp(-1.0, 1.0)
        if bool((t == 0).all()):
            return x0_hat
        eps_eff = (x - sqrt_a_t * x0_hat) / sqrt_one_minus_a_t
        sigma2 = ((1 - a_prev) / (1 - a_t)).clamp(min=0.0) \
            * (1 - a_t / a_prev).clamp(min=0.0)
        dir_coef = (1 - a_prev - sigma2).clamp(min=0.0).sqrt()
        return (
            a_prev.sqrt() * x0_hat
            + dir_coef * eps_eff
            + sigma2.sqrt() * torch.randn_like(x)
        )

    gauss_diffusion.p_sample_plms = types.MethodType(eta1_step, gauss_diffusion)


def run_timing(engine: ENUNU,step=None):

    # USTファイルを編集する
    ust = utaupy.ust.load(engine.path_ust)
    ust = engine.edit_ust(ust)
    ust.write(engine.path_ust)

    # UST → LAB の変換をする
    logging.info('Converting UST -> LAB')
    enulib.utauplugin2score.utauplugin2score(
        engine.path_ust,
        engine.path_table,
        engine.path_full_score,
        strict_sinsy_style=False,
    )

    
    if step is None:
        # フルラベルファイルを読み取る
        logging.info('Loading LAB')
        labels = hts.load(engine.path_full_score)

        # LABファイルを編集する。
        labels = engine.edit_score(labels)

        engine.svs_timing(
            labels=labels,
            dtype=np.float32,
            vocoder_type='auto',
            post_filter_type='gv',
            force_fix_vuv=True,
            segmented_synthesis=True,
        )
        duration_modified_labels = hts.load(engine.path_full_timing).round_()
        duration_modified_labels = engine.edit_timing(duration_modified_labels)
    else:
        engine.path_full_timing = engine.path_full_score
        duration_modified_labels = hts.load(engine.path_full_timing).round_()
        with open(engine.path_mono_timing, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(duration_modified_labels)))
        duration_modified_labels = engine.edit_timing(duration_modified_labels,"timing_editor_2")

def run_acoustic(engine: ENUNU,kind='linear',editor_f0=None,style_shift=0):
    engine.svs_acoustic(
            dtype=np.float32,
            vocoder_type='auto',
            post_filter_type='gv',
            force_fix_vuv=True,
            segmented_synthesis=False,
            kind=kind,
            editor_f0=editor_f0,
            style_shift=style_shift,
        )

def run_pitch(engine: ENUNU,style_shift=0):
    return engine.svs_pitch(
            style_shift=style_shift,
            vocoder_type='auto',
            post_filter_type='gv',
        )


def run_npy(engine: ENUNU,kind='linear'):
    print(f'{datetime.now()} : pitch interpolation mode : {kind}')
    engine.svs_npy(
            vocoder_type='auto',
            kind=kind,
        )

def run_synthesizer(out_wav_path: str,engine: ENUNU):
    
    # フルラベルファイルを読み取る
    logging.info('Loading LAB')
    labels = hts.load(engine.path_full_score)

    # WAVファイル出力
    wav_data,sample_rate = engine.svs_synthe(
        labels=labels,
        dtype=np.int16,
        vocoder_type='auto',
        post_filter_type='gv',
        force_fix_vuv=True,
    )

    wav_data = adjust_wav_gain_for_float32(wav_data)
    wavfile.write(out_wav_path, rate=sample_rate, data=wav_data)

def setup(path_plugin: str):
    # 引用符を削除
    path_plugin = path_plugin.strip('"\'')
    # USTの形式のファイルでなければエラー
    if not path_plugin.endswith('.tmp') or path_plugin.endswith('.ust'):
        raise ValueError('Input file must be UST or TMP(plugin).')
    # UTAUの一時ファイルに書いてある設定を読み取る
    logging.info('reading settings in TMP')
    path_ust, voice_dir, _ = get_project_path(path_plugin)

    # 日付時刻を取得
    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # tkinterの親Windowを表示させないようにする
    root = tkinter.Tk()
    root.withdraw()
    # 入出力パスを設定する
    if path_ust is not None:
        songname = splitext(basename(path_ust))[0]
        out_dir = dirname(path_ust)
        temp_dir = join(out_dir, f'{songname}_enutemp')
    # WAV出力パス指定なしかつUST未保存の場合
    else:
        logging.info('USTが保存されていないのでデスクトップにWAV出力します。')
        songname = f'temp__{str_now}'
        out_dir = mkdtemp(prefix='enunu-')
        temp_dir = join(out_dir, f'{songname}_enutemp')

    makedirs(temp_dir, exist_ok=True)

    ## NNSVS / ENUNU モデルを探す
    # model フォルダ
    if packed_model_exists(join(voice_dir, 'model')):
        model_dir = join(voice_dir, 'model')
    # 直置き
    elif packed_model_exists(voice_dir):
        model_dir = voice_dir
    # ENUNU<1.0.0 向けのディレクトリ構成
    elif exists(join(voice_dir, 'enuconfig.yaml')):
        logger.info('Regacy ENUNU model is selected. Converting it for the compatibility...')
        model_dir = join(voice_dir, 'model')
        makedirs(model_dir, exist_ok=True)
        print('----------------------------------------------')
        wrapped_enunu2nnsvs(voice_dir, model_dir)
        print('\n----------------------------------------------')
        logger.info('Converted.')

    # configファイルがあるか調べて、なければ例外処理
    else:
        raise Exception('UTAU音源選択でENUNU用モデルを指定してください。')
    assert model_dir

    # カレントディレクトリを音源フォルダに変更する
    chdir(voice_dir)
    # モデルを読み取る
    logging.info('Loading models')
    engine = ENUNU(model_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
    engine.set_paths(temp_dir=temp_dir)

    # NOTE: 後方互換のため
    # enuconfigが存在する場合、そこに記載されている拡張機能のパスをconfigに追加する
    if exists(join(voice_dir, 'enuconfig.yaml')):
        with open(join(voice_dir, 'enuconfig.yaml'), encoding='utf-8') as f:
            enuconfig = yaml.safe_load(f)
        engine.config['extensions'] = enuconfig.get('extensions')
        del enuconfig

    # USTを一時フォルダに複製
    logger.info(f'{datetime.now()} : copying UST')
    if engine.path_ust is not None:
        shutil.copy2(path_plugin, engine.path_ust)
    else:
        raise ValueError("Engine path_ust is None")
    # Tableファイルを一時フォルダに複製
    # Tableファイルの場所はモデルの場所から探す
    logger.info(f'{datetime.now()} : copying Table')
    if engine.path_table is None:
        # shutil.copy2(find_table(model_dir), engine.path_table)
        engine.path_table = find_table(model_dir)
    else:
        raise ValueError("Engine path_table is None")

    return engine

def updete_path(path_plugin: str,engine: ENUNU):
    # 引用符を削除
    path_plugin = path_plugin.strip('"\'')
    # USTの形式のファイルでなければエラー
    if not path_plugin.endswith('.tmp') or path_plugin.endswith('.ust'):
        raise ValueError('Input file must be UST or TMP(plugin).')
    # UTAUの一時ファイルに書いてある設定を読み取る
    logging.info('reading settings in TMP')
    path_ust, voice_dir, _ = get_project_path(path_plugin)

    # 日付時刻を取得
    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # tkinterの親Windowを表示させないようにする
    root = tkinter.Tk()
    root.withdraw()
    # 入出力パスを設定する
    if path_ust is not None:
        songname = splitext(basename(path_ust))[0]
        out_dir = dirname(path_ust)
        temp_dir = join(out_dir, f'{songname}_enutemp')
    # WAV出力パス指定なしかつUST未保存の場合
    else:
        logging.info('USTが保存されていないのでデスクトップにWAV出力します。')
        songname = f'temp__{str_now}'
        out_dir = mkdtemp(prefix='enunu-')
        temp_dir = join(out_dir, f'{songname}_enutemp')

    makedirs(temp_dir, exist_ok=True)

    engine.set_paths(temp_dir=temp_dir, path_feedback=path_plugin)
    
    # USTを一時フォルダに複製
    logger.info(f'{datetime.now()} : copying UST')
    if engine.path_ust is not None:
        shutil.copy2(path_plugin, engine.path_ust)
    else:
        raise ValueError("Engine path_ust is None")

    return temp_dir

# ↑EnunuServerCustom


def main(path_plugin: str, path_wav: str | None = None, play_wav: bool = False) -> str:
    """
    UTAUプラグインのファイルから音声を生成する
    """
    # 引用符を削除
    path_plugin = path_plugin.strip('"\'')
    if path_wav is not None:
        path_wav = path_wav.strip('"\'')

    # USTの形式のファイルでなければエラー
    if not (path_plugin.endswith('.tmp') or path_plugin.endswith('.ust')):
        raise ValueError('Input file must be UST or TMP(plugin).')
    # UTAUの一時ファイルに書いてある設定を読み取る
    logger.info('reading settings in TMP')
    path_ust, voice_dir, _ = get_project_path(path_plugin)

    # 日付時刻を取得
    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # wav出力パスが指定されていない(プラグインとして実行している)場合
    if path_wav is None:
        # tkinterの親Windowを表示させないようにする
        root = tkinter.Tk()
        root.withdraw()
        # 入出力パスを設定する
        if path_ust is not None:
            songname = splitext(basename(path_ust))[0]
            out_dir = dirname(path_ust)
            temp_dir = join(out_dir, f'{songname}_enutemp')
        # WAV出力パス指定なしかつUST未保存の場合
        else:
            logging.info('USTが保存されていないのでデスクトップにWAV出力します。')
            songname = f'temp__{str_now}'
            out_dir = mkdtemp(prefix='enunu-')
            temp_dir = join(out_dir, f'{songname}_enutemp')

    # WAV出力パスが指定されている場合
    else:
        songname = splitext(basename(path_wav))[0]
        out_dir = dirname(path_wav)
        temp_dir = join(out_dir, f'{songname}_enutemp')
        path_wav = abspath(path_wav)

    ## NNSVS / ENUNU モデルを探す
    # model フォルダ
    if packed_model_exists(join(voice_dir, 'model')):
        model_dir = join(voice_dir, 'model')
    # 直置き
    elif packed_model_exists(voice_dir):
        model_dir = voice_dir
    # ENUNU<1.0.0 向けのディレクトリ構成
    elif exists(join(voice_dir, 'enuconfig.yaml')):
        logger.info('Regacy ENUNU model is selected. Converting it for the compatibility...')
        model_dir = join(voice_dir, 'model')
        makedirs(model_dir, exist_ok=True)
        print('----------------------------------------------')
        wrapped_enunu2nnsvs(voice_dir, model_dir)
        print('\n----------------------------------------------')
        logger.info('Converted.')

    # configファイルがあるか調べて、なければ例外処理
    else:
        raise Exception('UTAU音源選択でENUNU用モデルを指定してください。')
    assert model_dir

    # カレントディレクトリを音源フォルダに変更する
    chdir(voice_dir)

    # 一時フォルダを作成する
    makedirs(temp_dir, exist_ok=True)

    # モデルを読み取る
    logger.info('Loading models')
    engine = ENUNU(model_dir)
    engine.set_paths(temp_dir=temp_dir, path_feedback=path_plugin)

    # NOTE: 後方互換のため
    # enuconfigが存在する場合、そこに記載されている拡張機能のパスをconfigに追加する
    if exists(join(voice_dir, 'enuconfig.yaml')):
        with open(join(voice_dir, 'enuconfig.yaml'), encoding='utf-8') as f:
            enuconfig = yaml.safe_load(f)
        engine.config['extensions'] = enuconfig.get('extensions')
        del enuconfig

    # USTを一時フォルダに複製
    logger.info(f'{datetime.now()} : copying UST')
    shutil.copy2(path_plugin, engine.path_ust)
    # Tableファイルを一時フォルダに複製
    logger.info(f'{datetime.now()} : copying Table')
    shutil.copy2(find_table(model_dir), engine.path_table)

    # USTファイルを編集する
    ust = utaupy.ust.load(engine.path_ust)
    ust = engine.edit_ust(ust)
    ust.write(engine.path_ust)

    # UST → LAB の変換をする
    logging.info('Converting UST -> LAB')
    enulib.utauplugin2score.utauplugin2score(
        engine.path_ust,
        engine.path_table,
        engine.path_full_score,
        strict_sinsy_style=False,
    )

    # フルラベルファイルを読み取る
    logging.info('Loading LAB')
    labels = hts.load(engine.path_full_score)

    # LABファイルを編集する。
    labels = engine.edit_score(labels)

    # 音声を生成する
    # NOTE: engine.svs を分解してタイミング補正を行えるように改造中。
    logging.info('Generating WAV')
    wav_data, sample_rate = engine.svs(
        labels,
        dtype=np.float32,
        vocoder_type='auto',
        post_filter_type='gv',
        force_fix_vuv=True,
        segmented_synthesis=SEGMENTED_SYNTHESIS,
    )

    # wav出力のフォーマットを確認する
    wav_data = adjust_wav_gain_for_float32(wav_data)

    # WAV出力先が未定の場合
    if path_wav is None:
        print(
            '表示されているエクスプローラーの画面から、WAVファイルに名前を付けて保存してください。'
        )
        if out_dir is not None:
            initialdir = out_dir
        else:
            initialdir = expanduser(join('~', 'Desktop'))
        # wavファイルの保存先を指定
        path_wav = asksaveasfilename(
            initialdir=initialdir,
            initialfile=f'{songname}.wav',
            filetypes=[('Wave sound file', '.wav'), ('All files', '*')],
            defaultextension='.wav',
        )
    assert path_wav != '', 'ファイル名が入力されていません'

    # wav出力
    wavfile.write(path_wav, rate=sample_rate, data=wav_data)

    # 音声を再生する。
    if exists(path_wav) and play_wav is True:
        startfile(path_wav)  # noqa: S606

    return path_wav


if __name__ == '__main__':
    logging.debug('sys.argv: %s', sys.argv)
    if len(sys.argv) == 1:
        # コマンドライン引数が指定されていない場合は、TMPファイルを指定する。
        main(input('Input file path of TMP(plugin)\n>>> '), path_wav=None, play_wav=True)
    else:
        # コマンドライン引数を取得する。
        parser = ArgumentParser()
        parser.add_argument('ust', type=str, help='Input file path (UST or TMP)')
        parser.add_argument('--wav', type=str, required=False, help='Output file path (WAV)')
        parser.add_argument('--play', action='store_true', help='Play WAV after rendering or not')
        args = parser.parse_args()
        # 実行
        main(args.ust, path_wav=args.wav, play_wav=args.play)
