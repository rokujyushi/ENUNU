namespace OnnxSvs;

/// <summary>timelag / duration の推論の設定。既定値は nnsvs の実際の動作 (predict_timing の既定値と、ForceClipInputFeatures 以外は同じ)。</summary>
public sealed record TimingOptions
{
    public bool LogF0Conditioning { get; init; } = true;
    /// <remarks>
    /// nnsvs.gen の既定は true だが、nnsvs.svs が渡す scaler (nnsvs.util.MinMaxScaler) は、gen 側の
    /// isinstance(sklearn の MinMaxScaler) の判定に当たらず、実際にはクリップされない。それに合わせて false を既定にする。
    /// </remarks>
    public bool ForceClipInputFeatures { get; init; } = false;
    public (double Min, double Max) AllowedRange { get; init; } = (-20, 20);
    public (double Min, double Max) AllowedRangeRest { get; init; } = (-40, 40);
    /// <summary>1 フレームの長さ (ms)。</summary>
    public int FramePeriod { get; init; } = 5;

    public long FrameShift => (long)(FramePeriod * 1e4);
}

/// <summary>音素長の予測。確率モデルなら Sigma2 (フレーム²) も入る。</summary>
public sealed record DurationPrediction(double[] Mean, double[]? Sigma2);

/// <summary>
/// nnsvs.gen の predict_timelag / predict_duration / postprocess_duration / predict_timing を移したもの。
/// 動的特徴量 (MLPG が要るモデル) は未対応。
/// </summary>
public static class TimingPredictor
{
    private static bool IsSilence(string label)
    {
        var isFullContext = label.Contains('@');
        return isFullContext ? label.Contains("-sil") || label.Contains("-pau") : label is "sil" or "pau";
    }

    private static void RequireStatic(ModelStage stage)
    {
        if (stage.Config.AnyDynamicFeatures)
        {
            throw new NotSupportedException($"{stage.Name}: 動的特徴量 (MLPG) を使うモデルは未対応です");
        }
    }

    /// <summary>音符ごとの time-lag (100ns 単位)。labels は 5 ms 単位に丸めたものを渡す。</summary>
    public static double[] PredictTimelag(HtsLabel labels, QuestionSet qs, ModelStage timelag, TimingOptions? options = null)
    {
        options ??= new TimingOptions();
        RequireStatic(timelag);
        var noteIndices = Conditioning.GetNoteIndices(labels);
        var noteLabels = new HtsLabel(noteIndices.Select(i => labels.Lines[i]));
        var pitchIndices = Conditioning.GetPitchIndices(qs);

        var features = LinguisticFeatures.PhoneLevel(noteLabels, qs);
        var x = Conditioning.Prepare(features, timelag, pitchIndices, options.LogF0Conditioning,
            options.ForceClipInputFeatures);
        var output = timelag.Run(x);
        var pred = timelag.Out.InverseTransform(output.Y);

        var result = new double[pred.Length];
        for (var i = 0; i < pred.Length; i++)
        {
            var lag = Math.Round(pred[i][0], MidpointRounding.ToEven);
            var range = IsSilence(noteLabels.Lines[i].Context) ? options.AllowedRangeRest : options.AllowedRange;
            result[i] = Math.Clamp(lag, range.Min, range.Max) * options.FrameShift;
        }
        return result;
    }

    /// <summary>音素ごとの長さ (フレーム)。</summary>
    public static DurationPrediction PredictDuration(HtsLabel labels, QuestionSet qs, ModelStage duration,
        TimingOptions? options = null)
    {
        options ??= new TimingOptions();
        RequireStatic(duration);
        var pitchIndices = Conditioning.GetPitchIndices(qs);
        var features = LinguisticFeatures.PhoneLevel(labels, qs);
        var x = Conditioning.Prepare(features, duration, pitchIndices, options.LogF0Conditioning,
            options.ForceClipInputFeatures);
        var output = duration.Run(x);
        var mean = duration.Out.InverseTransform(output.Y).Select(r => r[0]).ToArray();

        if (output.Sigma == null)
        {
            for (var i = 0; i < mean.Length; i++)
            {
                if (mean[i] <= 0)
                {
                    mean[i] = 1;
                }
                mean[i] = Math.Round(mean[i], MidpointRounding.ToEven);
            }
            return new DurationPrediction(mean, null);
        }

        var outScaler = (StandardScaler)duration.Out;
        var sigma2 = output.Sigma.Select(r => Math.Max((double)r[0] * r[0] * outScaler.Var[0], 1e-14)).ToArray();
        return new DurationPrediction(mean, sigma2);
    }

    /// <summary>
    /// time-lag を使って音符ごとに音素長を整え、ラベルの時刻を作り直す (Ref: arXiv 2108.02776)。
    /// </summary>
    public static HtsLabel PostprocessDuration(HtsLabel labels, DurationPrediction pred, double[] lag,
        TimingOptions? options = null)
    {
        options ??= new TimingOptions();
        long fs = options.FrameShift;
        var noteIndices = Conditioning.GetNoteIndices(labels);
        noteIndices.Add(labels.Count);

        var output = new List<HtsLabelLine>();
        for (var i = 1; i < noteIndices.Count; i++)
        {
            int a = noteIndices[i - 1], b = noteIndices[i];
            var count = b - a;
            var first = labels.Lines[a];

            // 音符の長さ (フレーム)。nnmnkwii の duration_features は int に切り捨てる
            var L = (int)((first.EndTime - first.StartTime) / (double)fs);
            var lHat = i < noteIndices.Count - 1
                ? L - (lag[i - 1] - lag[i]) / fs
                : L - lag[i - 1] / fs;
            lHat = Math.Max(lHat, 1);

            // 音符の開始時刻を time-lag で動かす
            var start = (double)first.StartTime + lag[i - 1];
            start = Math.Min(start, (double)labels.Lines[a].EndTime - fs * count);
            start = Math.Max(start, 0);
            if (output.Count > 0)
            {
                start = Math.Max(start, output[^1].StartTime + fs);
            }
            // Python の p.start_times は、ここまでで最初の音素の開始時刻だけが使われる

            var mu = pred.Mean[a..b];
            double[] dNorm;
            if (pred.Sigma2 == null)
            {
                var sum = mu.Sum();
                dNorm = mu.Select(d => lHat * d / sum).ToArray();
            }
            else
            {
                var sigma2 = pred.Sigma2[a..b];
                var rho = (lHat - mu.Sum()) / sigma2.Sum();
                dNorm = mu.Select((m, k) => m + rho * sigma2[k]).ToArray();
                if (dNorm.Any(d => d <= 0))
                {
                    var muSum = mu.Sum();
                    dNorm = mu.Select(m => lHat * m / muSum).ToArray();
                }
            }
            dNorm = dNorm.Select(d => Math.Round(d, MidpointRounding.ToEven)).Select(d => d <= 0 ? 1 : d).ToArray();

            // 音素長から開始/終了時刻を作る (set_durations は 5 ms 固定)
            var offset = (long)start;
            long end = 0;
            var lines = new List<HtsLabelLine>();
            double cumulative = 0;
            long begin = offset;
            for (var k = 0; k < count; k++)
            {
                cumulative += dNorm[k] * HtsLabel.DefaultFrameShift;
                end = offset + (long)cumulative;
                lines.Add(new HtsLabelLine(begin, end, labels.Lines[a + k].Context));
                begin = end;
            }

            if (output.Count > 0)
            {
                output[^1] = output[^1] with { EndTime = lines[0].StartTime };
            }
            output.AddRange(lines);
        }
        return new HtsLabel(output);
    }

    /// <summary>timelag + duration + postprocess_duration。labels は 5 ms 単位に丸めたもの。</summary>
    public static HtsLabel PredictTiming(HtsLabel labels, QuestionSet qs, ModelStage timelag, ModelStage duration,
        TimingOptions? options = null)
    {
        options ??= new TimingOptions();
        var lag = PredictTimelag(labels, qs, timelag, options);
        var durations = PredictDuration(labels, qs, duration, options);
        return PostprocessDuration(labels, durations, lag, options);
    }
}
