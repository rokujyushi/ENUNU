namespace OnnxSvs;

/// <summary>acoustic 推論の設定。既定値は nnsvs.gen.predict_acoustic と同じ (クリップは TimingOptions と同じ事情で false)。</summary>
public sealed record AcousticOptions
{
    public bool LogF0Conditioning { get; init; } = true;
    public bool ForceClipInputFeatures { get; init; } = false;
    /// <summary>推論の前に、音高をこの分 (cent) だけずらす。</summary>
    public double F0ShiftInCent { get; init; } = 0;
    public int FramePeriod { get; init; } = 5;

    public long FrameShift => (long)(FramePeriod * 1e4);
}

/// <summary>
/// nnsvs.gen.predict_acoustic を移したもの。ラベル → 言語特徴量 (コーステコーディング) → モデル →
/// 逆変換 → (動的特徴があれば) MLPG で、フレームごとの静的な音響特徴 (ストリームを連結したもの) を返す。
/// </summary>
public static class AcousticPredictor
{
    /// <param name="labels">時刻が音素ごとにそろった (5 ms 単位に丸めた) ラベル</param>
    public static double[][] PredictAcoustic(HtsLabel labels, QuestionSet qs, ModelStage acoustic,
        AcousticOptions? options = null)
    {
        options ??= new AcousticOptions();
        var pitchIndices = Conditioning.GetPitchIndices(qs);
        var features = LinguisticFeatures.FrameLevelCoarseCoding(labels, qs, options.FrameShift);

        var x = Conditioning.Prepare(features, acoustic, pitchIndices, options.LogF0Conditioning,
            options.ForceClipInputFeatures, options.F0ShiftInCent);
        var output = acoustic.Run(x);
        var config = acoustic.Config;
        var outScaler = (StandardScaler)acoustic.Out;

        if (output.Sigma != null)
        {
            var mu = acoustic.Out.InverseTransform(output.Y);
            if (!config.AnyDynamicFeatures)
            {
                return mu;
            }
            var sigma2 = output.Sigma
                .Select(row => row.Select((s, d) => Math.Max((double)s * s * outScaler.Var[d], 1e-14)).ToArray())
                .ToArray();
            return Mlpg.MultiStream(mu, sigma2, Mlpg.GetWindows(config.NumWindows), config.StreamSizes,
                config.HasDynamicFeatures);
        }

        var pred = acoustic.Out.InverseTransform(output.Y);
        if (!config.AnyDynamicFeatures)
        {
            return pred;
        }
        return Mlpg.MultiStream(pred, pred.Select(_ => outScaler.Var).ToArray(),
            Mlpg.GetWindows(config.NumWindows), config.StreamSizes, config.HasDynamicFeatures);
    }
}
