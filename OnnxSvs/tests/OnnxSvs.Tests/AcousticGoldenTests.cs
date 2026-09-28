using System.Text.Json;
using Xunit;

namespace OnnxSvs.Tests;

/// <summary>nnsvs.gen.predict_acoustic (golden_acoustic.json) と突き合わせる。行は一定間隔で抜き出したもの。</summary>
public class AcousticGoldenTests
{
    private static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "data", name);

    [Theory]
    [InlineData("mdn", 0)]
    [InlineData("mdn", 100)]
    [InlineData("det", 0)]
    [InlineData("det", 100)]
    public void PredictAcoustic_MatchesNnsvs(string variant, int shift)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Data("golden_acoustic.json")));
        var g = doc.RootElement.GetProperty(variant).GetProperty($"shift{shift}");
        using var stage = ModelStage.Load(Data(Path.Combine("acoustic_models", variant)), "acoustic");
        var labels = HtsLabel.Load(Data("sample_full.lab")).Rounded();
        var qs = QuestionSet.Load(Data("jp_qst001_nnsvs.hed"));

        var y = AcousticPredictor.PredictAcoustic(labels, qs, stage, new AcousticOptions { F0ShiftInCent = shift });

        Assert.Equal(g.GetProperty("frames").GetInt32(), y.Length);
        Assert.Equal(g.GetProperty("dim").GetInt32(), y[0].Length);
        var rows = g.GetProperty("rows").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var expected = g.GetProperty("y").EnumerateArray().ToArray();
        for (var i = 0; i < rows.Length; i++)
        {
            var e = expected[i].EnumerateArray().Select(x => x.GetDouble()).ToArray();
            for (var d = 0; d < e.Length; d++)
            {
                Assert.InRange(y[rows[i]][d] - e[d], -1e-3, 1e-3);
            }
        }
    }
}
