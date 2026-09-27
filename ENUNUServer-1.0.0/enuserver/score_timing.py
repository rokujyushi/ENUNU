"""UST → 楽譜ラベル (score.full) の時刻を、UST の tick から丸めずに計算する。

utaupy はノート長を 96 分音符 (20 tick) 単位に丸めてから (`round(length / 20)`)、それを積み上げて
ラベルの時刻を決める。OpenUtau は 1 音素を 1 ノートにして 16 tick や 109 tick のような長さで送るので、
ノートごとに最大 ±10 tick の誤差が出て、フレーズの後ろほど楽譜からずれる
(37 小節からの 115 音素のフレーズで、最後の「ん」が楽譜より 53 ms 早くなっていた)。

ノート長の文脈 (e8, 96 分音符単位) はモデルの学習時と同じ形にしたいので丸めたまま残し、
音素の開始・終了時刻だけを正確な tick から計算し直す。
"""
from decimal import ROUND_HALF_UP, Decimal
from itertools import chain

_applied = False


def exact_reset_time(song, ust_notes):
    """utaupy.hts.Song.reset_time と同じ割り当てを、丸める前の UST のノート長で行う。"""
    notes = song.all_notes
    if len(notes) != len(ust_notes):
        # 1 UST ノート = 1 HTS ノートでなければ、utaupy の時刻のまま
        return
    t_start = Decimal(0)
    t_end = Decimal(0)
    for note, ust_note in zip(notes, ust_notes):
        # 1 tick = 60 / (tempo * 480) 秒 = 1250000 / tempo (100ns 単位)
        t_end += Decimal(1250000 * int(ust_note.length)) / Decimal(ust_note.tempo)
        for phoneme in chain.from_iterable(note):
            phoneme.start = t_start.quantize(Decimal('0'), rounding=ROUND_HALF_UP)
            phoneme.end = t_end.quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        t_start = t_end


def apply():
    """utaupy.utils.ustobj2songobj を、時刻を丸めない版に差し替える (何度呼んでもよい)。"""
    global _applied
    if _applied:
        return
    _applied = True

    import utaupy.utils
    from utaupy.utils import _ust2hts

    original = _ust2hts.ustobj2songobj

    def ustobj2songobj(ust, d_table, key_of_the_note=None):
        song = original(ust, d_table, key_of_the_note=key_of_the_note)
        exact_reset_time(song, ust.notes)
        return song

    # enulib.utauplugin2score は utaupy.utils.ustobj2songobj の形で呼ぶので、両方差し替える
    _ust2hts.ustobj2songobj = ustobj2songobj
    utaupy.utils.ustobj2songobj = ustobj2songobj
