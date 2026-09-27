"""テスト共通の設定とヘルパー。

実行方法 (ENUNUServer-1.0.0 フォルダで):
    python-3.13.15-embed-amd64\\python.exe -m unittest discover -s ..\\tests -v

実モデルを使うテストは、音源フォルダが見つからなければスキップする。
音源は環境変数 ENUNU_TEST_VOICES (; 区切り) で変更できる。
"""
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(REPO_DIR, 'ENUNUServer-1.0.0')
# テストを動かしている組み込み Python でサーバーも起動する
PYTHON_EXE = sys.executable
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

DEFAULT_VOICES = [
    r'F:\UTAU\voice\ENUNU_KanadeShia_20251101',            # world 拡散 (lf0 分離) + 拡張機能あり
    r'F:\UTAU\voice\ENUNU_欲音ルコ♀_melf0-diffusion_V100',  # melf0 拡散
]


def test_voices():
    env = os.environ.get('ENUNU_TEST_VOICES')
    voices = [v for v in env.split(';') if v] if env else DEFAULT_VOICES
    return [v for v in voices if os.path.isdir(v)]


def write_tmp(path, voice, notes, tempo=120):
    """OpenUtau の EnunuUtils.WriteUst と同じ形式の tmp を書く。

    notes: [(lyric, length, notenum), ...]。OpenUtau と同じく 1 ノート = 1 音素
    (テストでは母音だけを使う)。前後に休符を付ける。
    """
    notes = [('R', 240, 60)] + list(notes) + [('R', 240, 60)]
    lines = ['[#SETTING]', f'Tempo={tempo}', 'Tracks=1', f'Project={path}',
             f'VoiceDir={voice}', f'CacheDir={os.path.dirname(path)}', 'Mode2=True']
    for i, (lyric, length, notenum) in enumerate(notes):
        lines += [f'[#{i}]', f'Lyric={lyric}', f'Length={length}', f'NoteNum={notenum}', 'Velocity=100']
    lines.append('[#TRACKEND]')
    with open(path, 'w', encoding='cp932') as f:
        f.write('\n'.join(lines) + '\n')
    return path


PHRASE_A = [('あ', 480, 62), ('い', 480, 64), ('あ', 960, 65)]
PHRASE_B = [('お', 480, 67), ('え', 480, 65), ('お', 960, 60)]
