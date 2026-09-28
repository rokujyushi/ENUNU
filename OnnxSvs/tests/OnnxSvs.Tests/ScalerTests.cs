using Xunit;

namespace OnnxSvs.Tests;

public class ScalerTests
{
    [Fact]
    public void MinMax_IsScaleTimesXPlusMin_AsInNnsvs()
    {
        var s = new MinMaxScaler(new[] { -1f, 2f }, new[] { 0.5f, 4f });
        Assert.Equal(new[] { 0.5f * 3 - 1, 4f * 1 + 2 }, s.Transform(new[] { new[] { 3f, 1f } })[0]);
        Assert.Equal(new[] { 3f, 1f }, s.InverseTransform(s.Transform(new[] { new[] { 3f, 1f } }))[0]);
    }

    [Fact]
    public void Standard_IsMeanAndScale()
    {
        var s = new StandardScaler(new[] { 1f, 2f }, new[] { 2f, 4f });
        Assert.Equal(new[] { 1f, 1f }, s.Transform(new[] { new[] { 3f, 6f } })[0]);
        Assert.Equal(new[] { 3f, 6f }, s.InverseTransform(new[] { new[] { 1f, 1f } })[0]);
    }
}
