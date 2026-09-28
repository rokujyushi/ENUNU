using System.Text.Json;
using Xunit;

namespace OnnxSvs.Tests;

/// <summary>nnmnkwii.paramgen.mlpg / nnsvs.multistream.multi_stream_mlpg の出力 (golden_mlpg.json) と突き合わせる。</summary>
public class MlpgGoldenTests
{
    private static JsonElement Case(string name)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "data", "golden_mlpg.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty(name).Clone();
    }

    private static double[][] Matrix(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray();

    private static void AssertClose(double[][] expected, double[][] actual, double tol)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var t = 0; t < expected.Length; t++)
        {
            Assert.Equal(expected[t].Length, actual[t].Length);
            for (var d = 0; d < expected[t].Length; d++)
            {
                Assert.InRange(actual[t][d] - expected[t][d], -tol, tol);
            }
        }
    }

    [Theory]
    [InlineData("30x4")]
    [InlineData("5x2")]
    [InlineData("2x1")]
    public void Generate_MatchesNnmnkwii(string name)
    {
        var c = Case(name);
        var mean = Matrix(c.GetProperty("mean"));
        var y = Matrix(c.GetProperty("y"));
        var windows = Mlpg.GetWindows(mean[0].Length / y[0].Length);
        AssertClose(y, Mlpg.Generate(mean, Matrix(c.GetProperty("var")), windows), 1e-5);
    }

    [Fact]
    public void MultiStream_MatchesNnsvs()
    {
        var c = Case("multistream");
        var sizes = c.GetProperty("stream_sizes").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        var dyn = c.GetProperty("has_dynamic").EnumerateArray().Select(x => x.GetBoolean()).ToArray();
        var v = c.GetProperty("var").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var mean = Matrix(c.GetProperty("mean"));
        var variances = mean.Select(_ => v).ToArray();
        var result = Mlpg.MultiStream(mean, variances, Mlpg.GetWindows(3), sizes, dyn);
        AssertClose(Matrix(c.GetProperty("y")), result, 1e-5);
    }
}
