using Xunit;

namespace OnnxSvs.Tests;

public class ScalerTests
{
    [Fact]
    public void MinMax_IsScaleTimesXPlusMin_AsInNnsvs()
    {
        var s = new MinMaxScaler(new[] { -1.0, 2.0 }, new[] { 0.5, 4.0 });
        var t = s.Transform(new[] { new[] { 3f, 1f } });
        Assert.Equal(new[] { 0.5f * 3 - 1, 4f * 1 + 2 }, t[0]);
        Assert.Equal(new[] { 3.0, 1.0 }, s.InverseTransform(t)[0]);
    }

    [Fact]
    public void Standard_IsMeanAndScale()
    {
        var s = new StandardScaler(new[] { 1.0, 2.0 }, new[] { 4.0, 16.0 }, new[] { 2.0, 4.0 });
        var t = s.Transform(new[] { new[] { 3f, 6f } });
        Assert.Equal(new[] { 1f, 1f }, t[0]);
        Assert.Equal(new[] { 3.0, 6.0 }, s.InverseTransform(t)[0]);
    }
}
