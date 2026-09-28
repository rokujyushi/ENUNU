using Xunit;

namespace OnnxSvs.Tests;

public class ConditioningTests
{
    [Fact]
    public void Interp1d_FillsGapsLinearly_AndExtendsEnds()
    {
        // 両端と途中に 0 がある
        var f = Conditioning.Interp1d(new double[] { 0, 0, 2, 0, 0, 5, 0 });
        Assert.Equal(new double[] { 2, 2, 2, 3, 4, 5, 5 }, f);
    }

    [Fact]
    public void Interp1d_AllZero_IsUnchanged()
    {
        Assert.Equal(new double[] { 0, 0, 0 }, Conditioning.Interp1d(new double[] { 0, 0, 0 }));
    }

    [Fact]
    public void ConditionPitchColumn_UsesLogHz_AndInterpolatesRests()
    {
        // MIDI 69 = 440 Hz。0 は音高なし
        var features = new[] { new[] { 69f }, new[] { 0f }, new[] { 69f } };
        Conditioning.ConditionPitchColumn(features, 0);
        var expected = (float)Math.Log(440.0);
        Assert.All(features, row => Assert.Equal(expected, row[0]));
    }

    [Fact]
    public void GetNoteIndices_SplitsWhereStartTimeChanges()
    {
        var labels = new HtsLabel(new[]
        {
            new HtsLabelLine(0, 100, "a"), new HtsLabelLine(0, 100, "b"),
            new HtsLabelLine(100, 200, "c"), new HtsLabelLine(200, 300, "d"),
            new HtsLabelLine(200, 300, "e"),
        });
        Assert.Equal(new[] { 0, 2, 3 }, Conditioning.GetNoteIndices(labels));
    }

    [Fact]
    public void GetPitchIndices_ListsLeadingPitchQuestions()
    {
        var qs = QuestionSet.Parse(new[]
        {
            "QS \"q1\" {a}",
            "CQS \"d\" {/D:(\\NOTE)!}",
            "CQS \"e\" {/E:(\\NOTE)]}",
            "CQS \"x\" {/X:(\\d+)!}",
        });
        Assert.Equal(new[] { 1, 2 }, Conditioning.GetPitchIndices(qs));
    }
}
