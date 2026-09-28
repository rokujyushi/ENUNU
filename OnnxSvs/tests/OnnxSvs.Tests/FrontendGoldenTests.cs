using System.Text.Json;
using Xunit;

namespace OnnxSvs.Tests;

/// <summary>nnmnkwii の出力 (tests/tools/gen_golden.py で作った golden_frontend.json) と突き合わせる。</summary>
public class FrontendGoldenTests
{
    private static string Data(string name) => Path.Combine(AppContext.BaseDirectory, "data", name);

    private static JsonElement Golden(string hed)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Data("golden_frontend.json")));
        return doc.RootElement.GetProperty(hed).Clone();
    }

    private static float[] Row(JsonElement e) => e.EnumerateArray().Select(x => (float)x.GetDouble()).ToArray();

    public static IEnumerable<object[]> Heds => new[]
    {
        new object[] { "jp_qst001_nnsvs.hed" },
        new object[] { "jp_dev_latest.hed" },
    };

    [Theory]
    [MemberData(nameof(Heds))]
    public void PhoneLevel_MatchesNnmnkwii(string hed)
    {
        var g = Golden(hed);
        var qs = QuestionSet.Load(Data(hed));
        var labels = HtsLabel.Load(Data("sample_full.lab")).Rounded();

        Assert.Equal(g.GetProperty("dim").GetInt32(), qs.Dimension);
        var actual = LinguisticFeatures.PhoneLevel(labels, qs);
        var expected = g.GetProperty("phone").EnumerateArray().Select(Row).ToArray();

        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Theory]
    [MemberData(nameof(Heds))]
    public void FrameLevel_CoarseCoding_MatchesNnmnkwii(string hed)
    {
        var g = Golden(hed);
        var qs = QuestionSet.Load(Data(hed));
        var labels = HtsLabel.Load(Data("sample_full.lab")).Rounded();
        var dim = qs.Dimension;

        var actual = LinguisticFeatures.FrameLevelCoarseCoding(labels, qs);

        var perPhone = g.GetProperty("frames_per_phone").EnumerateArray().Select(x => x.GetInt32()).ToArray();
        Assert.Equal(perPhone.Sum(), actual.Length);
        Assert.Equal(labels.NumFrames(), actual.Length);

        // 末尾 4 列 (位置 3 つ + 音素長) は全フレームで一致
        var cc = g.GetProperty("cc").EnumerateArray().Select(Row).ToArray();
        for (var f = 0; f < cc.Length; f++)
        {
            for (var k = 0; k < 4; k++)
            {
                Assert.Equal(cc[f][k], actual[f][dim + k], 6);
            }
        }

        // 一部のフレームは全次元が一致
        foreach (var prop in g.GetProperty("rows").EnumerateObject())
        {
            Assert.Equal(Row(prop.Value), actual[int.Parse(prop.Name)]);
        }
    }

    [Fact]
    public void Rounded_RoundsHalfToEven()
    {
        var labels = new HtsLabel(new[]
        {
            new HtsLabelLine(25000, 75000, "a"),    // 0.5 -> 0, 1.5 -> 2
            new HtsLabelLine(125000, 175000, "b"),  // 2.5 -> 2, 3.5 -> 4
        }).Rounded();
        Assert.Equal(new long[] { 0, 100000, 100000, 200000 },
            labels.Lines.SelectMany(l => new[] { l.StartTime, l.EndTime }).ToArray());
    }
}
