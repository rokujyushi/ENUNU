namespace OnnxSvs;

public enum PostFilter
{
    None,
    /// <summary>nnsvs の "gv" (音符のフレームだけ、スペクトルの分散を学習データに合わせる)。</summary>
    Gv,
}

/// <summary>nnsvs.gen.postprocess_acoustic の設定。既定値も同じ (WORLD の特徴量のみ対応)。</summary>
public sealed record PostprocessOptions
{
    public int FramePeriod { get; init; } = 5;
    public bool RelativeF0 { get; init; } = false;
    public PostFilter PostFilter { get; init; } = PostFilter.Gv;
    public bool TrajectorySmoothing { get; init; } = true;
    public double TrajectorySmoothingCutoff { get; init; } = 50;
    public double TrajectorySmoothingCutoffF0 { get; init; } = 20;
    public double VuvThreshold { get; init; } = 0.5;
    /// <summary>後処理の最後に log-f0 へ足す音高のずれ (cent)。</summary>
    public double F0ShiftInCent { get; init; } = 0;
    public bool ForceFixVuv { get; init; } = false;
    public bool FillSilenceToRest { get; init; } = false;
    /// <summary>モデルが出すビブラートのストリーム (5・6 番目) を f0 に反映するときの深さの倍率。0 でビブラートなし。</summary>
    public double VibratoScale { get; init; } = 1.0;

    public long FrameShift => (long)(FramePeriod * 1e4);
}

/// <summary>WORLD の特徴量。Lf0 と Vuv は 1 フレーム 1 値。</summary>
public sealed record WorldParams(double[][] Mgc, double[] Lf0, double[] Vuv, double[][] Bap)
{
    /// <summary>
    /// フレームごとの音高のずれ (cent) を log-f0 に足したものを返す。UST のビブラートなど、
    /// 外で作った f0 の変化を後から重ねるための口。有声かどうかは Vuv で決まるので、無声の値も足してよい。
    /// </summary>
    public WorldParams WithF0DeltaCents(double[] deltaCents)
    {
        if (deltaCents.Length != Lf0.Length)
        {
            throw new ArgumentException($"長さがフレーム数 ({Lf0.Length}) と違います");
        }
        var lf0 = Lf0.Select((v, t) => v + deltaCents[t] * Math.Log(2) / 1200).ToArray();
        return this with { Lf0 = lf0 };
    }
}

/// <summary>
/// nnsvs.gen.postprocess_acoustic (feature_type="world") を移したもの。GV、ストリームの分割、
/// vuv の補正、f0 の組み立て、休符埋め、軌跡のなめらか化までを行う。
/// ビブラートのストリーム (差分方式・正弦波方式) に対応。merlin / 学習済みの post-filter は未対応。
/// </summary>
public static class AcousticPostprocess
{
    /// <summary>acoustic の静的特徴のうち、学習データの分散 (GV 用) を静的な次元だけ取り出す (nnsvs.util.extract_static_scaler)。</summary>
    public static double[] StaticVariance(double[] fullVariance, ModelConfig config)
    {
        var result = new List<double>();
        var start = 0;
        for (var s = 0; s < config.StreamSizes.Length; s++)
        {
            var size = config.StreamSizes[s];
            var staticSize = config.HasDynamicFeatures[s] ? size / config.NumWindows : size;
            result.AddRange(fullVariance[start..(start + staticSize)]);
            start += size;
        }
        return result.ToArray();
    }

    public static int[] StaticStreamSizes(ModelConfig config)
        => config.StreamSizes
            .Select((size, s) => config.HasDynamicFeatures[s] ? size / config.NumWindows : size).ToArray();

    /// <param name="acoustic">PredictAcoustic の出力 (静的特徴を連結したもの)</param>
    /// <param name="staticVariance">学習データの静的特徴の分散 (StaticVariance で作る)</param>
    public static WorldParams Postprocess(double[][] acoustic, HtsLabel labels, QuestionSet qs,
        int[] staticStreamSizes, double[] staticVariance, PostprocessOptions? options = null)
    {
        options ??= new PostprocessOptions();
        if (staticStreamSizes.Length is < 4 or > 6)
        {
            throw new NotSupportedException("ストリームは 4 つ (mgc, lf0, vuv, bap)、ビブラートの差分方式なら 5 つ、正弦波方式なら 6 つです");
        }
        var T = acoustic.Length;
        var features = LinguisticFeatures.FrameLevelCoarseCoding(labels, qs, options.FrameShift);
        if (features.Length != T)
        {
            throw new ArgumentException($"音響特徴のフレーム数 ({T}) がラベルのフレーム数 ({features.Length}) と違います");
        }
        var pitchIdx = PitchIndex(qs);
        var x = acoustic.Select(r => (double[])r.Clone()).ToArray();

        if (options.PostFilter == PostFilter.Gv)
        {
            var noteFrames = Enumerable.Range(0, T).Where(t => features[t][pitchIdx] > 0).ToArray();
            VarianceScaling(staticVariance[..staticStreamSizes[0]], x, staticStreamSizes[0], 2, noteFrames);
        }

        // ストリームに分ける
        var mgcDim = staticStreamSizes[0];
        var bapDim = staticStreamSizes[3];
        var mgc = x.Select(r => r[..mgcDim]).ToArray();
        var target = x.Select(r => r[mgcDim]).ToArray();
        var vuv = x.Select(r => r[mgcDim + 1]).ToArray();
        var bap = x.Select(r => r[(mgcDim + 2)..(mgcDim + 2 + bapDim)]).ToArray();

        var vibStart = mgcDim + 2 + bapDim;
        double[][]? vib = staticStreamSizes.Length >= 5
            ? x.Select(r => r[vibStart..(vibStart + staticStreamSizes[4])]).ToArray() : null;
        double[]? vibFlags = staticStreamSizes.Length == 6
            ? x.Select(r => r[vibStart + staticStreamSizes[4]]).ToArray() : null;

        if (options.ForceFixVuv)
        {
            vuv = CorrectVuvByPhone(vuv, qs, features);
        }

        // f0 (Hz。無声は 0) → log-f0 を補間
        var f0 = new double[T];
        double[]? score = options.RelativeF0 ? Conditioning.ScoreLogF0(features, pitchIdx) : null;
        for (var t = 0; t < T; t++)
        {
            var lf0 = score != null ? target[t] + score[t] : target[t];
            f0[t] = vuv[t] < options.VuvThreshold || lf0 == 0 ? 0 : Math.Exp(lf0);
        }
        if (vib != null)
        {
            f0 = vibFlags != null
                ? Vibrato.GenSineVibrato(f0, (int)(1 / (options.FramePeriod * 0.001)),
                    vib.Select((r, t) => vibFlags[t] < 0.5 ? 0 : r[0]).ToArray(),
                    vib.Select((r, t) => vibFlags[t] < 0.5 ? 0 : r[1]).ToArray(), options.VibratoScale)
                : f0.Select((v, t) => v + options.VibratoScale * vib[t][0]).ToArray();
        }
        var lf0Filled = Conditioning.Interp1d(f0.Select(v => v != 0 ? Math.Log(v) : 0).ToArray());

        if (options.FillSilenceToRest)
        {
            FillSilence(mgc, bap, NonRestSoftMask(qs, features, pitchIdx));
        }

        if (options.F0ShiftInCent != 0)
        {
            var offset = options.F0ShiftInCent * Math.Log(2) / 1200;
            lf0Filled = lf0Filled.Select(v => v + offset).ToArray();
        }

        if (options.TrajectorySmoothing)
        {
            var modfs = (int)(1 / (options.FramePeriod * 0.001));
            lf0Filled = Dsp.LowpassFilter(lf0Filled, modfs, options.TrajectorySmoothingCutoffF0);
            SmoothColumns(mgc, modfs, options.TrajectorySmoothingCutoff);
            SmoothColumns(bap, modfs, options.TrajectorySmoothingCutoff);
        }
        if (bapDim <= 5)
        {
            foreach (var row in bap)
            {
                for (var d = 0; d < row.Length; d++)
                {
                    row[d] = Math.Clamp(row[d], -60, 0);
                }
            }
        }
        return new WorldParams(mgc, lf0Filled, vuv, bap);
    }

    /// <summary>ModelStage (acoustic) の設定と scaler から、PredictAcoustic の出力に後処理をかける。</summary>
    public static WorldParams Postprocess(double[][] acoustic, HtsLabel labels, QuestionSet qs,
        ModelStage acousticStage, PostprocessOptions? options = null)
    {
        var config = acousticStage.Config;
        var variance = StaticVariance(((StandardScaler)acousticStage.Out).Var, config);
        return Postprocess(acoustic, labels, qs, StaticStreamSizes(config), variance, options);
    }

    /// <summary>nnsvs.io.hts.get_pitch_index: /E で始まる最初の数値特徴 (音符の音高) の列番号。</summary>
    public static int PitchIndex(QuestionSet qs)
    {
        for (var idx = 0; idx < qs.Numeric.Count; idx++)
        {
            if (qs.Numeric[idx].Pattern.ToString().StartsWith("/E"))
            {
                return qs.Binary.Count + idx;
            }
        }
        return qs.Binary.Count;
    }

    /// <summary>nnsvs.postfilters.variance_scaling。note_frames のフレームだけ、次元 offset 以降を変える。</summary>
    internal static void VarianceScaling(double[] gv, double[][] feats, int dims, int offset, int[] noteFrames)
    {
        if (noteFrames.Length == 0)
        {
            return;
        }
        var n = noteFrames.Length;
        var mean = new double[dims];
        var variance = new double[dims];
        for (var d = 0; d < dims; d++)
        {
            foreach (var t in noteFrames)
            {
                mean[d] += feats[t][d];
            }
            mean[d] /= n;
            foreach (var t in noteFrames)
            {
                var diff = feats[t][d] - mean[d];
                variance[d] += diff * diff;
            }
            variance[d] /= n;
        }
        foreach (var t in noteFrames)
        {
            for (var d = offset; d < dims; d++)
            {
                feats[t][d] = Math.Sqrt(gv[d] / variance[d]) * (feats[t][d] - mean[d]) + mean[d];
            }
        }
    }

    private static int[] BinaryColumns(QuestionSet qs, params string[] names)
        => Enumerable.Range(0, qs.Binary.Count)
            .Where(k => names.Any(n => qs.Binary[k].Name.Contains(n, StringComparison.Ordinal))).ToArray();

    /// <summary>nnsvs.gen.correct_vuv_by_phone: hed の C-VUV_Voiced / C-VUV_Unvoiced と、休符・息の音素で vuv を決める。</summary>
    internal static double[] CorrectVuvByPhone(double[] vuv, QuestionSet qs, float[][] features)
    {
        var result = (double[])vuv.Clone();
        // nnsvs は最初に見つかった C-VUV_Voiced の列が 0 のとき「なし」と扱う (idx > 0)
        var voiced = BinaryColumns(qs, "C-VUV_Voiced");
        if (voiced.Length > 0 && voiced[0] > 0)
        {
            for (var t = 0; t < result.Length; t++)
            {
                if (features[t][voiced[0]] > 0)
                {
                    result[t] = 1.0;
                }
            }
        }
        foreach (var col in BinaryColumns(qs, "C-VUV_Unvoiced"))
        {
            for (var t = 0; t < result.Length; t++)
            {
                if (features[t][col] > 0)
                {
                    result[t] = 0.0;
                }
            }
        }
        foreach (var col in BinaryColumns(qs, "C-Phone_sil", "C-Phone_pau", "C-Phone_br"))
        {
            for (var t = 0; t < result.Length; t++)
            {
                if (features[t][col] > 0)
                {
                    result[t] = 0.0;
                }
            }
        }
        return result;
    }

    /// <summary>nnsvs.gen._get_nonrest_frame_soft_mask: 1 秒より長い休符を 0、それ以外を 1 にして、なめらかにしたもの。</summary>
    internal static double[] NonRestSoftMask(QuestionSet qs, float[][] features, int pitchIdx,
        int winLength = 200, double durationThreshold = 1.0)
    {
        var T = features.Length;
        var mask = Enumerable.Repeat(1.0, T).ToArray();
        var silCols = BinaryColumns(qs, "C-Phone_sil", "C-Phone_pau");
        if (silCols.Length == 0)
        {
            return mask;
        }
        var durIdx = -1;
        for (var k = 0; k < qs.Numeric.Count; k++)
        {
            if (qs.Numeric[k].Name.Contains("e7", StringComparison.Ordinal))
            {
                durIdx = k;
                break;
            }
        }
        if (durIdx < 0)
        {
            throw new InvalidDataException("休符埋めには、音符の長さの数値特徴 (名前に e7 を含む) が hed に必要です");
        }
        foreach (var col in silCols)
        {
            for (var t = 0; t < T; t++)
            {
                if (features[t][col] > 0 && features[t][qs.Binary.Count + durIdx] * 0.01 > durationThreshold)
                {
                    mask[t] = 0;
                }
            }
        }
        // scipy.signal.convolve(mask, ones(win)/win, mode="same")
        var smooth = new double[T];
        var start = (winLength - 1) / 2;
        for (var t = 0; t < T; t++)
        {
            var sum = 0.0;
            for (var j = 0; j < winLength; j++)
            {
                var i = t + start - j;
                if (i >= 0 && i < T)
                {
                    sum += mask[i];
                }
            }
            smooth[t] = sum / winLength;
        }
        for (var t = 0; t < T; t++)
        {
            if (features[t][pitchIdx] > 0)
            {
                smooth[t] = 1.0;
            }
        }
        return smooth;
    }

    /// <summary>nnsvs.gen._fill_silence_to_world_params: 休符のフレームを無音のおおよその値に近づける。</summary>
    internal static void FillSilence(double[][] mgc, double[][] bap, double[] mask)
    {
        for (var t = 0; t < mgc.Length; t++)
        {
            for (var d = 0; d < mgc[t].Length; d++)
            {
                var sil = d switch { 0 => -23.3, 1 => 0.0679, 2 => 0.00640, _ => 1e-3 };
                mgc[t][d] = mgc[t][d] * mask[t] + (1 - mask[t]) * sil;
            }
            for (var d = 0; d < bap[t].Length; d++)
            {
                bap[t][d] = bap[t][d] * mask[t] + (1 - mask[t]) * 1e-11;
            }
        }
    }

    private static void SmoothColumns(double[][] rows, int fs, double cutoff)
    {
        var dims = rows[0].Length;
        for (var d = 0; d < dims; d++)
        {
            var filtered = Dsp.LowpassFilter(rows.Select(r => r[d]).ToArray(), fs, cutoff);
            for (var t = 0; t < rows.Length; t++)
            {
                rows[t][d] = filtered[t];
            }
        }
    }
}
