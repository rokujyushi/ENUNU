using System.Text.Json;
using Xunit;

namespace OnnxSvs.Tests;

/// <summary>pysptk.mc2sp / mcepalpha、pyworld.get_cheaptrick_fft_size、nnsvs.gen.gen_world_params (golden_world.json) と突き合わせる。</summary>
public class WorldGoldenTests
{
    private static JsonElement Golden()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "data", "golden_world.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.Clone();
    }

    private static double[][] Matrix(JsonElement e) =>
        e.EnumerateArray().Select(r => r.EnumerateArray().Select(x => x.GetDouble()).ToArray()).ToArray();

    private static double[] Vector(JsonElement e) => e.EnumerateArray().Select(x => x.GetDouble()).ToArray();

    [Fact]
    public void McepAlpha_MatchesPysptk()
    {
        foreach (var p in Golden().GetProperty("alpha").EnumerateObject())
        {
            Assert.Equal(p.Value.GetDouble(), Sptk.McepAlpha(int.Parse(p.Name)), 9);
        }
    }

    [Fact]
    public void CheapTrickFftSize_MatchesPyworld()
    {
        foreach (var p in Golden().GetProperty("fft_size").EnumerateObject())
        {
            Assert.Equal(p.Value.GetInt32(), WorldVocoder.CheapTrickFftSize(int.Parse(p.Name)));
        }
    }

    [Fact]
    public void Mc2Sp_MatchesPysptk()
    {
        foreach (var c in Golden().GetProperty("mc2sp").EnumerateArray())
        {
            var alpha = c.GetProperty("alpha").GetDouble();
            var fftLen = c.GetProperty("fftlen").GetInt32();
            var mc = Matrix(c.GetProperty("mc"));
            var expected = Matrix(c.GetProperty("log_sp"));
            for (var t = 0; t < mc.Length; t++)
            {
                var sp = Sptk.Mc2Sp(mc[t], alpha, fftLen);
                Assert.Equal(expected[t].Length, sp.Length);
                for (var k = 0; k < sp.Length; k++)
                {
                    Assert.InRange(Math.Log(sp[k]) - expected[t][k], -1e-8, 1e-8);
                }
            }
        }
    }

    [Fact]
    public void GenWorldParams_MatchesNnsvs()
    {
        var g = Golden().GetProperty("gen_world_params");
        var p = new WorldParams(Matrix(g.GetProperty("mgc")), Vector(g.GetProperty("lf0")),
            Vector(g.GetProperty("vuv")), Matrix(g.GetProperty("bap")));
        var options = new WorldSynthesisOptions
        {
            SampleRate = g.GetProperty("fs").GetInt32(), VuvThreshold = g.GetProperty("vuv_threshold").GetDouble(),
        };

        var (f0, sp, ap) = WorldVocoder.GenWorldParams(p, options);

        var expectedF0 = Vector(g.GetProperty("f0"));
        var expectedSp = Matrix(g.GetProperty("log_sp"));
        var expectedAp = Matrix(g.GetProperty("ap"));
        Assert.Equal(expectedF0.Length, f0.Length);
        for (var t = 0; t < f0.Length; t++)
        {
            Assert.InRange(f0[t] - expectedF0[t], -1e-9, 1e-9);
            for (var k = 0; k < sp[t].Length; k++)
            {
                Assert.InRange(Math.Log(sp[t][k]) - expectedSp[t][k], -1e-8, 1e-8);
                Assert.InRange(ap[t][k] - expectedAp[t][k], -1e-9, 1e-9);
            }
        }
    }
}

/// <summary>Worldline に渡す前の組み立て (codec と synthesizer は偽物で確認する)。</summary>
public class WorldVocoderTests
{
    private sealed class FakeCodec : IWorldCodec
    {
        public double[][] DecodeSpectralEnvelope(double[][] mgc, int fftSize, int fs)
            => mgc.Select(_ => Enumerable.Repeat(7.0, fftSize / 2 + 1).ToArray()).ToArray();

        public double[][] DecodeAperiodicity(double[][] bap, int fftSize, int fs)
            => bap.Select(row => Enumerable.Repeat(row[0], fftSize / 2 + 1).ToArray()).ToArray();
    }

    private sealed class FakeSynthesizer : IWorldSynthesizer
    {
        public (int FftSize, double FramePeriod, int Fs, double[] F0)? Called;

        public double[] Synthesize(double[] f0, double[][] sp, double[][] ap, int fftSize, double framePeriod, int fs)
        {
            Called = (fftSize, framePeriod, fs, f0);
            return new double[f0.Length * 240];
        }
    }

    [Fact]
    public void Codec_IsUsed_ForSpectrumAndApAndUnvoicedFramesAreFixed()
    {
        var p = new WorldParams(
            new[] { new[] { 0.0 }, new[] { 0.0 }, new[] { 0.0 } },
            new[] { Math.Log(200.0), Math.Log(300.0), Math.Log(400.0) },
            new[] { 1.0, 0.1, 1.0 },
            // 5 次元以下なので WORLD の codec で戻す。範囲外の値は 0 から 1 に収める
            new[] { new[] { 0.4, 0.0 }, new[] { 0.3, 0.0 }, new[] { -0.5, 0.0 } });
        var options = new WorldSynthesisOptions { SampleRate = 48000, UseWorldCodec = true };

        var (f0, sp, ap) = WorldVocoder.GenWorldParams(p, options, new FakeCodec());

        Assert.Equal(2048 / 2 + 1, sp[0].Length);
        Assert.All(sp[0], v => Assert.Equal(7.0, v));
        Assert.Equal(200.0, f0[0], 6);
        Assert.Equal(0.0, f0[1]);
        Assert.Equal(400.0, f0[2], 6);
        Assert.Equal(0.4, ap[0][0]);
        Assert.Equal(1.0, ap[1][0]);  // 無声
        Assert.Equal(0.0, ap[2][0]);  // -0.5 は 0 に
        Assert.Equal(0.3, ap[1][1]);
    }

    [Fact]
    public void Codec_IsRequired_WhenNeeded()
    {
        var p = new WorldParams(new[] { new[] { 0.0 } }, new[] { 5.0 }, new[] { 1.0 }, new[] { new[] { 0.0 } });
        Assert.Throws<InvalidOperationException>(() => WorldVocoder.GenWorldParams(p, new WorldSynthesisOptions()));
    }

    [Fact]
    public void Synthesize_PassesSettingsToSynthesizer()
    {
        var p = new WorldParams(new[] { new[] { 0.0 }, new[] { 0.0 } }, new[] { 5.0, 5.0 }, new[] { 1.0, 1.0 },
            new[] { new[] { 0.0 }, new[] { 0.0 } });
        var synth = new FakeSynthesizer();
        var options = new WorldSynthesisOptions { SampleRate = 44100, FramePeriod = 5, UseWorldCodec = true };

        var wav = WorldVocoder.Synthesize(p, options, synth, new FakeCodec());

        Assert.Equal(480, wav.Length);
        Assert.Equal(2048, synth.Called!.Value.FftSize);
        Assert.Equal(5, synth.Called.Value.FramePeriod);
        Assert.Equal(44100, synth.Called.Value.Fs);
        Assert.Equal(2, synth.Called.Value.F0.Length);
    }
}
