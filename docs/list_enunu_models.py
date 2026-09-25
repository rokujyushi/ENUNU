"""ENUNU 音源のモデル構造を調べて CSV に書き出すスクリプト。

使い方:
    py list_enunu_models.py [--out 出力CSV] [音源フォルダ ...]
    (フォルダ省略時は DEFAULT_DIRS を調べる)

対応する形式:
    - 旧形式: enuconfig.yaml + model_dir/{timelag,duration,acoustic,...}/model.yaml
    - 新形式 (NNSVS packed model): config.yaml + {timelag,duration,acoustic,vocoder}_model.yaml
      https://nnsvs.github.io/ の「packed model」
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path

import yaml

DEFAULT_DIRS = [
    r"F:\UTAU\voice",                              # OpenUtau / UTAU 音源
    r"I:\NEditor\N-Editor\resources\app\singer",  # N-Editor 同梱音源
]
DEFAULT_OUT = Path(__file__).with_name("enunu_models.csv")


def read_text(path):
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def load_yaml(path):
    if not path or not path.is_file():
        return None
    try:
        return yaml.safe_load(read_text(path))
    except Exception:
        return None


def short(target):
    """nnsvs.acoustic_models.Foo -> Foo"""
    return target.rsplit(".", 1)[-1] if target else ""


def describe_net(model_yaml):
    """netG の _target_ と、マルチストリームならサブモデル構成を返す。"""
    if not model_yaml:
        return "", "", "", ""
    net = model_yaml.get("netG") or model_yaml.get("generator") or {}
    target = net.get("_target_", "")
    subs = []
    for key, val in net.items():
        if isinstance(val, dict) and "_target_" in val:
            subs.append(f"{key}={short(val['_target_'])}")
    hparams = []
    for key in ("hidden_dim", "num_layers", "num_lstm_layers", "lstm_hidden_dim",
                "num_gaussians", "reduction_factor", "bidirectional"):
        if key in net:
            hparams.append(f"{key}={net[key]}")
    dims = f"{net.get('in_dim', '')}->{net.get('out_dim', '')}" if "in_dim" in net else ""
    return target, " / ".join(subs), " ".join(hparams), dims


def model_kind(target):
    """NNSVS のモデルクラス名から大まかな種類を判定する。"""
    t = target or ""
    rules = [
        ("Diffusion", "Diffusion"),
        ("NPSSMDN", "Multistream (NPSS-MDN)"),
        ("NPSSMultistream", "Multistream (NPSS)"),
        ("Multistream", "Multistream"),
        ("NonAttentive", "Non-attentive decoder"),
        ("ResSkipF0FFConvLSTM", "ResSkipF0FFConvLSTM"),
        ("ResF0", "ResF0 (F0 residual)"),
        ("Conv1dResnetMDN", "Conv1dResnet + MDN"),
        ("RMDN", "RMDN (MDN)"),
        ("MDN", "MDN"),
        ("Conv1dResnet", "Conv1dResnet"),
        ("LSTMRNN", "LSTM-RNN"),
        ("FFN", "Feed-forward"),
    ]
    for key, label in rules:
        if key in t:
            return label
    return short(t)


def vocoder_kind(target, feature_type):
    t = target or ""
    if not t:
        return "WORLD (信号処理)" if feature_type in ("", "world", None) else f"なし ({feature_type})"
    for key, label in (("ParallelHnUSFGAN", "Parallel HN-uSFGAN"), ("HnUSFGAN", "HN-uSFGAN"),
                       ("USFGAN", "uSFGAN"), ("SiFiGAN", "SiFi-GAN"), ("HiFiGAN", "HiFi-GAN"),
                       ("ParallelWaveGAN", "Parallel WaveGAN"), ("MelGAN", "MelGAN")):
        if key.lower() in t.lower():
            return label
    return short(t)


def hed_dims(hed_path):
    """question ファイルの QS / CQS 数を数える。"""
    if not hed_path or not hed_path.is_file():
        return ""
    text = read_text(hed_path)
    qs = len(re.findall(r"^\s*QS\s", text, re.M))
    cqs = len(re.findall(r"^\s*CQS\s", text, re.M))
    return f"QS={qs} CQS={cqs}"


def character_name(root):
    for p in (root / "character.txt", root.parent / "character.txt"):
        if p.is_file():
            m = re.search(r"^name\s*=\s*(.+)$", read_text(p), re.M)
            if m:
                return m.group(1).strip()
    return ""


def singer_type(root):
    for p in (root / "character.yaml", root.parent / "character.yaml"):
        y = load_yaml(p)
        if isinstance(y, dict) and y.get("singer_type"):
            return str(y["singer_type"])
    return ""


def extensions_summary(ext):
    if not isinstance(ext, dict):
        return ""
    parts = []
    for key, val in ext.items():
        if key in ("styles", "style_format") or val in (None, "built-in"):
            continue
        items = val if isinstance(val, list) else [val]
        names = [Path(str(v)).stem for v in items if v]
        if names:
            parts.append(f"{key}:{'+'.join(names)}")
    return " ; ".join(parts)


def pth_size_mb(paths):
    total = sum(p.stat().st_size for p in paths if p and p.is_file())
    return f"{total / 1024 / 1024:.1f}" if total else ""


def analyze_packed(folder, root):
    cfg = load_yaml(root / "config.yaml") or {}
    row = base_row(folder, root, "新形式 (packed model / config.yaml)")
    row["sample_rate"] = cfg.get("sample_rate", "")
    row["frame_period"] = cfg.get("frame_period", "")
    feature_type = cfg.get("feature_type", "world")
    row["feature_type"] = feature_type
    row["use_world_codec"] = cfg.get("use_world_codec", "")
    row["relative_f0"] = (cfg.get("acoustic") or {}).get("relative_f0", "")
    row["post_filter"] = (cfg.get("acoustic") or {}).get("post_filter_type", "") or \
        ("postfilter_model" if (root / "postfilter_model.pth").is_file() else "")
    row["question"] = "qst.hed" if (root / "qst.hed").is_file() else ""
    row["question_size"] = hed_dims(root / "qst.hed")
    for stage in ("timelag", "duration", "acoustic"):
        fill_stage(row, stage, load_yaml(root / f"{stage}_model.yaml"))
    ac = load_yaml(root / "acoustic_model.yaml") or {}
    row["acoustic_stream_sizes"] = str(ac.get("stream_sizes", ""))
    voc = load_yaml(root / "vocoder_model.yaml")
    voc_target = ((voc or {}).get("generator") or {}).get("_target_", "")
    row["vocoder_type"] = vocoder_kind(voc_target, feature_type)
    row["vocoder_class"] = voc_target
    row["extensions"] = extensions_summary(cfg.get("extensions"))
    row["model_size_mb"] = pth_size_mb(list(root.glob("*_model.pth")))
    row["_acoustic_pth"] = root / "acoustic_model.pth"
    return row


def analyze_legacy(folder, root):
    cfg = load_yaml(root / "enuconfig.yaml") or {}
    row = base_row(folder, root, "旧形式 (enuconfig.yaml)")
    row["sample_rate"] = cfg.get("sample_rate", "")
    row["frame_period"] = cfg.get("frame_period", "")
    row["feature_type"] = "world"
    row["relative_f0"] = (cfg.get("acoustic") or {}).get("relative_f0", "")
    q = cfg.get("question_path") or ""
    row["question"] = Path(q).name if q else ""
    row["question_size"] = hed_dims(root / q) if q else ""
    model_dir = root / (cfg.get("model_dir") or "")
    pths = []
    missing = []
    for stage in ("timelag", "duration", "acoustic"):
        y = load_yaml(model_dir / stage / "model.yaml")
        fill_stage(row, stage, y)
        ckpt = (cfg.get(stage) or {}).get("checkpoint") or "best_loss.pth"
        pths.append(model_dir / stage / ckpt)
        if not (model_dir / stage / ckpt).is_file():
            onnx = list((model_dir / stage).glob("*.onnx")) if (model_dir / stage).is_dir() else []
            missing.append(f"{stage}/{ckpt}" + (" (onnxのみ有)" if onnx else ""))
        if stage == "acoustic" and y:
            row["acoustic_stream_sizes"] = str(y.get("stream_sizes", ""))
    row["_missing"] = missing
    # 旧形式で追加のモデル (postfilter / vocoder など) があれば記録
    extra = sorted(p.name for p in model_dir.iterdir()
                   if p.is_dir() and p.name not in ("timelag", "duration", "acoustic")
                   and any(p.iterdir())) \
        if model_dir.is_dir() else []
    post = (cfg.get("acoustic") or {}).get("post_filter_type") \
        or ("post_filter" if (cfg.get("acoustic") or {}).get("post_filter") else "")
    row["post_filter"] = post or ""
    voc_cfg = cfg.get("vocoder") or {}
    voc_ckpt = voc_cfg.get("checkpoint") if isinstance(voc_cfg, dict) else None
    voc_target = ""
    for name in extra:
        for cand in ("config.yml", "config.yaml", "model.yaml"):
            y = load_yaml(model_dir / name / cand)
            if isinstance(y, dict):
                gen = y.get("generator") or y.get("netG") or {}
                voc_target = voc_target or gen.get("_target_", "") or y.get("generator_type", "")
    row["vocoder_type"] = vocoder_kind(voc_target, "world") if (voc_ckpt or voc_target) \
        else "WORLD (信号処理)"
    row["vocoder_class"] = voc_target
    row["extra_models"] = " ".join(extra)
    row["extensions"] = extensions_summary(cfg.get("extensions"))
    row["model_size_mb"] = pth_size_mb(pths)
    row["_acoustic_pth"] = pths[-1]
    return row


def n_editor_summary(root):
    """N-Editor 専用設定 n_editor.yaml の要点 (声質ミックス軸 / タイミング補正)。"""
    y = load_yaml(root / "n_editor.yaml")
    if not isinstance(y, dict):
        return "", "", ""
    ch = y.get("character") or {}
    display = " ".join(str(v) for v in (ch.get("name"), ch.get("variant")) if v)
    mix = []
    for axis, conf in (y.get("voice_mix") or {}).items():
        if isinstance(conf, dict):
            labels = "/".join(map(str, conf.get("labels") or []))
            mix.append(f"{axis}[{labels}] max={conf.get('max', '')}")
    tc = y.get("timing_correct") or {}
    timing = ""
    if tc:
        timing = f"consonant_ratio={tc.get('consonant_ratio', '')}"
        if tc.get("use_consonant_list"):
            timing += " consonant_list=on"
    return display, " ; ".join(mix), timing


def model_fingerprint(path):
    """同一モデル判定用: サイズ + 先頭/末尾 4MB の SHA-1。"""
    if not path or not path.is_file():
        return ""
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(4 << 20))
        if size > 8 << 20:
            f.seek(-(4 << 20), 2)
            h.update(f.read())
    return h.hexdigest()[:12]


def fill_stage(row, stage, y):
    target, subs, hparams, dims = describe_net(y)
    row[f"{stage}_kind"] = model_kind(target)
    row[f"{stage}_class"] = target
    row[f"{stage}_dims"] = dims
    if stage == "acoustic":
        row["acoustic_submodels"] = subs
        row["acoustic_hparams"] = hparams


def base_row(folder, root, fmt):
    return {
        "source": str(folder.parent),
        "folder": folder.name,
        "model_root": str(root.relative_to(folder.parent)),
        "name": character_name(root),
        "singer_type": singer_type(root),
        "format": fmt,
    }


COLUMNS = [
    "source", "folder", "name", "singer_type", "format", "model_root",
    "sample_rate", "frame_period", "feature_type", "use_world_codec", "relative_f0",
    "question", "question_size",
    "timelag_kind", "timelag_class", "timelag_dims",
    "duration_kind", "duration_class", "duration_dims",
    "acoustic_kind", "acoustic_class", "acoustic_dims", "acoustic_submodels",
    "acoustic_hparams", "acoustic_stream_sizes",
    "post_filter", "vocoder_type", "vocoder_class", "extra_models",
    "extensions", "n_editor_display", "n_editor_voice_mix", "n_editor_timing",
    "model_size_mb", "acoustic_fingerprint", "same_model_as", "note",
]


def find_root(folder):
    """音源フォルダ内でモデル本体があるディレクトリを探す (2階層まで)。"""
    for depth in ("", "*/", "*/*/"):
        for name in ("config.yaml", "enuconfig.yaml"):
            for p in folder.glob(depth + name):
                if name == "enuconfig.yaml":
                    return p.parent, "legacy"
                if (p.parent / "acoustic_model.yaml").is_file():
                    return p.parent, "packed"
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    folders = []
    for d in map(Path, args.dirs):
        if d.is_dir():
            folders += sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: p.name.lower())
        else:
            print(f"skip (not found): {d}")

    rows = []
    for folder in folders:
        beep = load_yaml(folder / "beep.yaml")
        if isinstance(beep, dict) and beep.get("engine") == "beep":
            # N-Editor の疑似シンガー: NNSVS モデルを持たない
            rows.append({"source": str(folder.parent), "folder": folder.name,
                         "name": character_name(folder),
                         "format": "N-Editor 疑似シンガー (beep.yaml / NNSVS 非使用)",
                         "vocoder_type": "矩形波 BEEP",
                         "n_editor_display": n_editor_summary(folder)[0],
                         "note": "音響モデルなし。ノート音高で矩形波を鳴らすだけ"})
            continue
        root, kind = find_root(folder)
        is_enunu_name = "enunu" in folder.name.lower()
        if root is None:
            if is_enunu_name:
                rows.append({"source": str(folder.parent), "folder": folder.name,
                             "format": "未展開/不明",
                             "note": "モデルファイルが見つからない (" +
                                     ", ".join(p.name for p in folder.iterdir()) + ")"})
            continue
        row = analyze_packed(folder, root) if kind == "packed" else analyze_legacy(folder, root)
        row["n_editor_display"], row["n_editor_voice_mix"], row["n_editor_timing"] = \
            n_editor_summary(root)
        row["acoustic_fingerprint"] = model_fingerprint(row.get("_acoustic_pth"))
        notes = []
        if row.get("_missing"):
            notes.append("チェックポイント欠落: " + ", ".join(row["_missing"]))
        if row.get("sample_rate") not in ("", 22050, 24000, 44100, 48000):
            notes.append(f"sample_rate={row['sample_rate']} は非標準値 (設定ミスの可能性)")
        if root != folder:
            notes.append(f"モデルはサブフォルダ {root.relative_to(folder)} にある")
        if kind == "legacy" and row.get("singer_type") != "enunu":
            notes.append("character.yaml に singer_type: enunu の指定なし")
        row["note"] = " / ".join(notes)
        rows.append(row)

    # acoustic モデルが同一の音源同士を相互参照する
    by_fp = {}
    for r in rows:
        if r.get("acoustic_fingerprint"):
            by_fp.setdefault(r["acoustic_fingerprint"], []).append(r)
    for group in by_fp.values():
        for r in group:
            r["same_model_as"] = " ; ".join(
                f"{Path(o['source']).name}/{o['folder']}" for o in group if o is not r)

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
