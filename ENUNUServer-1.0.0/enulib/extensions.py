#!/usr/bin/env python3
# Copyright (c) 2022 oatsu
"""
ENUNUで外部ツールを呼び出すときに必要な関数とか
"""

import os
import runpy
import subprocess
import sys
import traceback
from os import getcwd
from os.path import abspath, dirname, exists, isfile, splitext
from sys import executable
from typing import Union

import utaupy


def merge_mono_time_change_to_full(path_mono_lab, path_full_lab):
    """モノラベルの時刻でフルラベルの時刻を上書きする。

    外部ソフトではフルラベルを加工せずに
    モノラベルだけ加工する場合が多いだろうから。
    """
    # モノラベルを読み取る
    mono_label = utaupy.label.load(path_mono_lab)
    # フルラベルを読み取る
    full_label = utaupy.label.load(path_full_lab)
    # 時刻を上書きする
    for ph_mono, ph_full in zip(mono_label, full_label):
        ph_full.start = ph_mono.start
        ph_full.end = ph_mono.end
    # フルラベルを上書き保存する
    full_label.write(path_full_lab)


def merge_full_time_change_to_mono(path_full_lab, path_mono_lab):
    """フルラベルの時刻でモノラベルの時刻を上書きする。
    """
    # 順番入れ替えただけ
    # pylint: disable=arguments-out-of-order
    merge_mono_time_change_to_full(path_full_lab, path_mono_lab)


def merge_mono_contexts_change_to_full(path_mono_lab, path_full_lab):
    """モノラベルの音素記号でフルラベルの音素記号を上書きする。
    フルラベル読み取りと保存の処理が遅いから出来たらやりたくない。
    """
    # モノラベルを読み取る
    mono_label = utaupy.label.load(path_mono_lab)
    # フルラベルを読み取る
    full_label = utaupy.hts.load(path_full_lab)
    # 音素を上書きする
    for ph_mono, ph_full in zip(mono_label, full_label):
        ph_full.phoneme.identity = ph_mono.symbol
    # フルラベルを上書き保存する
    full_label.write(path_full_lab)


def merge_full_contexts_change_to_mono(path_full_lab, path_mono_lab):
    """フルラベルの音素記号でモノラベルの音素記号を上書きする。
    こっちもフルラベル読み取りと保存の処理が遅いから出来たらやりたくない。
    """
    # モノラベルを読み取る
    mono_label = utaupy.label.load(path_mono_lab)
    # フルラベルを読み取る
    full_label = utaupy.hts.load(path_full_lab)
    # 音素を上書きする
    for ph_mono, ph_full in zip(mono_label, full_label):
        ph_mono.symbol = ph_full.phoneme.identity
    # フルラベルを上書き保存する
    mono_label.write(path_full_lab)


def str_has_been_changed(s_old: str, s_new: str):
    """モノラベルやフルラベルが変更されているか調べる。
    """
    return s_old.strip() != s_new.strip()


def parse_extension_path(path) -> Union[str, None]:
    """拡張機能のパス中のエイリアスを置換する。

    Following aliases are available
      - '%e' (the directory enunu.py exists in)
      - '%v' (the directory voicebank and enuconfig.yaml exists in)
      - '%u' (the directory utau.exe exists in)
    """
    if path is None:
        return None
    # 各種パスを取得
    voice_dir = getcwd()
    enunu_dir = dirname(dirname(__file__))
    # utau_dir = utaupy.utau.utau_root()
    # 置換
    path = path.replace(r'%e', enunu_dir)
    path = path.replace(r'%v', voice_dir)
    # path = path.replace(r'%u', utau_dir)
    return path


def run_extension(path=None, **kwargs):
    """
    USTやラベルを加工する外部ソフトを呼び出す。
    """
    # path = path.strip('"')
    if path is None:
        return None
    # パスに含まれるエイリアスを展開
    path = parse_extension_path(path)
    if not exists(path):
        raise ValueError(f'指定されたファイルが見つかりません。({path})')
    if not isfile(path):
        raise ValueError(f'指定されたパスはファイルではありません。({path})')

    # 拡張機能を呼び出すときのコマンド
    args = [path]
    # 辞書をコマンド用のリストに追加する。値がNoneだったら無視する。
    # kwargs = {'mono_score': path_mono_score, 'full_score': path_full_score}
    # ↓
    # --mono_score basename(path_mono_score) --full_score basename(path_full_score)
    for key, value in kwargs.items():
        if value is None:
            continue
        args.append(f'--{key}')
        args.append(value)

    # Pythonスクリプトはプロセス起動のコスト (1回 0.2 秒程度) を省くため、同じプロセス内で実行する。
    # ENUNU_EXTENSION_INPROCESS=0 で従来どおり subprocess で実行する。
    is_python = splitext(path.strip('"'))[1] == '.py'
    if is_python and os.environ.get('ENUNU_EXTENSION_INPROCESS', '1') != '0':
        try:
            run_python_extension_inprocess(path.strip('\'"'), args[1:])
            return
        except ImportError as e:
            # 同梱インタープリタの sys.modules と衝突する等。モジュール読み込み時点なのでファイルは未変更
            print(f'In-process extension failed to import ({e}). Falling back to subprocess.')

    # 拡張機能がPythonスクリプトな場合に、
    # ENUNU同梱のインタープリタで実行するようにコマンドを変更する。
    if is_python:
        args.insert(0, abspath(executable))

    # 拡張機能を呼び出す。
    subprocess.run(args, cwd=dirname(path.strip('\'"')), check=True)


def run_python_extension_inprocess(path, argv):
    """Pythonの拡張機能を subprocess と同じ条件 (argv / cwd / sys.path[0]) で、同じプロセス内で実行する。

    失敗時は subprocess.run(check=True) と同じく CalledProcessError を送出する。
    拡張機能が読み込んだ同じフォルダ内のモジュールは、音源ごとに同名のものがあり得るので実行後に破棄する。
    """
    script_dir = abspath(dirname(path))
    old_argv, old_cwd, old_path = sys.argv, getcwd(), list(sys.path)
    old_modules = set(sys.modules)
    sys.argv = [path] + list(argv)
    sys.path.insert(0, script_dir)
    os.chdir(script_dir)
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as e:
        if e.code not in (None, 0):
            raise subprocess.CalledProcessError(e.code if isinstance(e.code, int) else 1, [path] + list(argv)) from e
    except ImportError:
        raise
    except Exception as e:
        traceback.print_exc()
        raise subprocess.CalledProcessError(1, [path] + list(argv)) from e
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path
        os.chdir(old_cwd)
        for name in set(sys.modules) - old_modules:
            module_file = getattr(sys.modules.get(name), '__file__', None) or ''
            if abspath(module_file).startswith(script_dir + os.sep):
                del sys.modules[name]
