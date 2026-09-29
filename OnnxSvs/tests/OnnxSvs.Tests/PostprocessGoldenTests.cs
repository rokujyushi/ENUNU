using System.Text.Json;
using Xunit;

namespace OnnxSvs.Tests;

/// <summary>nnsvs.gen.postprocess_acoustic (golden_postprocess.json) と突き合わせる。行は一定間隔で抜き出したもの。</summary>
public class PostprocessGoldenTests
{
    private static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "data", name);

    private static double[][] Matrix(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray();

    private static double[] Vector(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    private static PostprocessOptions Options(string name) => name switch
    {
        "default" => new PostprocessOptions(),
        "fix_fill_shift" => new PostprocessOptions
        {
            ForceFixVuv = true, FillSilenceToRest = true, F0ShiftInCent = 50, VuvThreshold = 0.3,
        },
        "relative_plain" => new PostprocessOptions
        {
            RelativeF0 = true, PostFilter = PostFilter.None, TrajectorySmoothing = false,
        },
        "vib_sine" => new PostprocessOptions { VibratoScale = 1.5 },
        "vib_diff" => new PostprocessOptions { VibratoScale = 0.5, TrajectorySmoothing = false },
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [InlineData("default")]
    [InlineData("fix_fill_shift")]
    [InlineData("relative_plain")]
    [InlineData("vib_sine")]
    [InlineData("vib_diff")]
    public void Postprocess_MatchesNnsvs(string name)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Data("golden_postprocess.json")));
        var g = doc.RootElement;
        var c = g.GetProperty("cases").GetProperty(name);
        var labels = HtsLabel.Load(Data("sample_full.lab")).Rounded();
        var qs = QuestionSet.Load(Data("jp_qst001_nnsvs.hed"));
        var rows = g.GetProperty("rows").EnumerateArray().Select(x => x.GetInt32()).ToArray();

        // 入力は base (mgc, lf0, vuv, bap) の後ろに、ビブラートのストリームの列を足したもの
        var layout = c.GetProperty("input").GetString()!;
        var input = Matrix(g.GetProperty("inputs").GetProperty("base"));
        if (layout != "base")
        {
            var extra = Matrix(g.GetProperty("inputs").GetProperty(layout));
            input = input.Select((row, t) => row.Concat(extra[t]).ToArray()).ToArray();
        }
        var sizes = g.GetProperty("layouts").GetProperty(layout).EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var variance = Vector(g.GetProperty("static_var"))[..sizes.Sum()];

        var result = AcousticPostprocess.Postprocess(input, labels, qs, sizes, variance, Options(name));

        Assert.Equal(g.GetProperty("frames").GetInt32(), result.Lf0.Length);
        var mgc = Matrix(c.GetProperty("mgc"));
        var bap = Matrix(c.GetProperty("bap"));
        var lf0 = Vector(c.GetProperty("lf0"));
        var vuv = Vector(c.GetProperty("vuv"));
        for (var i = 0; i < rows.Length; i++)
        {
            var t = rows[i];
            for (var d = 0; d < mgc[i].Length; d++)
            {
                Assert.InRange(result.Mgc[t][d] - mgc[i][d], -2e-4, 2e-4);
            }
            for (var d = 0; d < bap[i].Length; d++)
            {
                Assert.InRange(result.Bap[t][d] - bap[i][d], -2e-4, 2e-4);
            }
            Assert.InRange(result.Lf0[t] - lf0[i], -1e-5, 1e-5);
            Assert.InRange(result.Vuv[t] - vuv[i], -1e-6, 1e-6);
        }
    }
}

public class WorldParamsTests
{
    [Fact]
    public void WithF0DeltaCents_ShiftsLog_F0ByCents()
    {
        var p = new WorldParams(new[] { new[] { 0.0 } }, new[] { Math.Log(200.0) }, new[] { 1.0 }, new[] { new[] { 0.0 } });
        var shifted = p.WithF0DeltaCents(new[] { 1200.0 });
        Assert.Equal(400.0, Math.Exp(shifted.Lf0[0]), 6);
        Assert.Throws<ArgumentException>(() => p.WithF0DeltaCents(new[] { 1.0, 2.0 }));
    }
}
