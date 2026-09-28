using System.Text.Json;
using Xunit;

namespace OnnxSvs.Tests;

/// <summary>
/// nnsvs.gen の predict_timelag / predict_duration / postprocess_duration の出力
/// (tests/tools/gen_golden_timing.py で作った golden_timing.json) と突き合わせる。
/// </summary>
public class TimingGoldenTests
{
    private static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "data", name);

    public static IEnumerable<object[]> Variants => new[]
    {
        new object[] { "det" },
        new object[] { "det_clip" },
        new object[] { "mdn" },
        new object[] { "mdn_clip" },
    };

    private sealed record Loaded(HtsLabel Labels, QuestionSet Qs, ModelStage Timelag, ModelStage Duration,
        TimingOptions Options, JsonElement Golden) : IDisposable
    {
        public void Dispose()
        {
            Timelag.Dispose();
            Duration.Dispose();
        }
    }

    private static Loaded Load(string variant)
    {
        var modelName = variant.Replace("_clip", "");
        var dir = Data(Path.Combine("timing_models", modelName));
        using var doc = JsonDocument.Parse(File.ReadAllText(Data("golden_timing.json")));
        return new Loaded(
            HtsLabel.Load(Data("sample_score.lab")).Rounded(),
            QuestionSet.Load(Data("jp_qst001_nnsvs.hed")),
            ModelStage.Load(dir, "timelag"),
            ModelStage.Load(dir, "duration"),
            new TimingOptions { ForceClipInputFeatures = variant.EndsWith("_clip") },
            doc.RootElement.GetProperty(variant).Clone());
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    [Theory]
    [MemberData(nameof(Variants))]
    public void Timelag_MatchesNnsvs(string variant)
    {
        using var t = Load(variant);
        var lag = TimingPredictor.PredictTimelag(t.Labels, t.Qs, t.Timelag, t.Options);
        Assert.Equal(Doubles(t.Golden.GetProperty("lag")), lag);
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void Duration_MatchesNnsvs(string variant)
    {
        using var t = Load(variant);
        var pred = TimingPredictor.PredictDuration(t.Labels, t.Qs, t.Duration, t.Options);
        var expected = Doubles(t.Golden.GetProperty("duration_mean"));
        Assert.Equal(expected.Length, pred.Mean.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], pred.Mean[i], 3);
        }
        if (t.Golden.TryGetProperty("duration_sigma2", out var sigma2Json))
        {
            var expectedSigma2 = Doubles(sigma2Json);
            Assert.NotNull(pred.Sigma2);
            for (var i = 0; i < expectedSigma2.Length; i++)
            {
                Assert.Equal(expectedSigma2[i], pred.Sigma2![i], expectedSigma2[i] * 1e-3 + 1e-9);
            }
        }
        else
        {
            Assert.Null(pred.Sigma2);
        }
    }

    [Theory]
    [MemberData(nameof(Variants))]
    public void PredictTiming_MatchesNnsvsLabels(string variant)
    {
        using var t = Load(variant);
        var result = TimingPredictor.PredictTiming(t.Labels, t.Qs, t.Timelag, t.Duration, t.Options);
        var expected = t.Golden.GetProperty("labels").EnumerateArray().ToArray();

        Assert.Equal(expected.Length, result.Count);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i][0].GetInt64(), result.Lines[i].StartTime);
            Assert.Equal(expected[i][1].GetInt64(), result.Lines[i].EndTime);
            Assert.Equal(expected[i][2].GetString(), result.Lines[i].Context);
        }
    }
}
