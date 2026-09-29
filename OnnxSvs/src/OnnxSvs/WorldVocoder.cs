namespace OnnxSvs;

/// <summary>WORLD の spectral envelope / aperiodicity の符号化 (WORLD の codec)。Worldline が提供する。</summary>
public interface IWorldCodec
{
    /// <summary>mgc (フレーム × 次元) を spectral envelope (フレーム × fftSize/2+1) にする。</summary>
    double[][] DecodeSpectralEnvelope(double[][] mgc, int fftSize, int fs);

    /// <summary>bap (フレーム × 帯域) を aperiodicity (フレーム × fftSize/2+1) にする。</summary>
    double[][] DecodeAperiodicity(double[][] bap, int fftSize, int fs);
}

/// <summary>WORLD の波形合成。f0 は Hz (無声は 0)、sp は power spectrum、ap は 0 から 1。</summary>
public interface IWorldSynthesizer
{
    double[] Synthesize(double[] f0, double[][] sp, double[][] ap, int fftSize, double framePeriod, int fs);
}

public sealed record WorldSynthesisOptions
{
    public int SampleRate { get; init; } = 48000;
    public double FramePeriod { get; init; } = 5;
    /// <summary>true なら mgc / bap を WORLD の codec で、false なら (SPTK の) メルケプストラムとして戻す。</summary>
    public bool UseWorldCodec { get; init; } = false;
    public double VuvThreshold { get; init; } = 0.5;
}

/// <summary>nnsvs.gen.gen_world_params と predict_waveform (vocoder_type="world") の WORLD 合成の前まで。</summary>
public static class WorldVocoder
{
    /// <summary>pyworld.get_cheaptrick_fft_size (f0_floor = 71 Hz)。</summary>
    public static int CheapTrickFftSize(int fs, double f0Floor = 71.0)
        => (int)Math.Pow(2.0, 1.0 + (int)(Math.Log(3.0 * fs / f0Floor + 1) / Math.Log(2.0)));

    public static (double[] F0, double[][] Sp, double[][] Ap) GenWorldParams(WorldParams p,
        WorldSynthesisOptions options, IWorldCodec? codec = null)
    {
        var fs = options.SampleRate;
        var fftLen = CheapTrickFftSize(fs);
        var T = p.Lf0.Length;

        double[][] sp;
        if (options.UseWorldCodec)
        {
            sp = Codec(codec).DecodeSpectralEnvelope(p.Mgc, fftLen, fs);
        }
        else
        {
            var alpha = Sptk.McepAlpha(fs);
            sp = p.Mgc.Select(row => Sptk.Mc2Sp(row, alpha, fftLen)).ToArray();
        }

        // bap が 5 次元より多いときは、メルケプストラムで表した aperiodicity (nnsvs の use_mcep_aperiodicity)
        double[][] ap;
        if (p.Bap[0].Length > 5)
        {
            var alpha = Sptk.McepAlpha(fs);
            ap = p.Bap.Select(row => Sptk.Mc2Sp(row, alpha, fftLen)).ToArray();
        }
        else
        {
            ap = Codec(codec).DecodeAperiodicity(p.Bap, fftLen, fs);
        }

        var f0 = new double[T];
        for (var t = 0; t < T; t++)
        {
            if (p.Vuv[t] < options.VuvThreshold)
            {
                // 無声のフレームは、aperiodicity の最初の点を 1 にしておく
                ap[t][0] = 1.0;
            }
            else if (p.Lf0[t] != 0)
            {
                f0[t] = Math.Exp(p.Lf0[t]);
            }
            // WORLD は範囲外の aperiodicity で壊れるので 0 から 1 に収める
            for (var k = 0; k < ap[t].Length; k++)
            {
                ap[t][k] = Math.Clamp(ap[t][k], 0.0, 1.0);
            }
        }
        return (f0, sp, ap);
    }

    /// <summary>WORLD で波形にする。値の範囲は nnsvs と同じ ([-1, 1] 程度の double)。</summary>
    public static double[] Synthesize(WorldParams p, WorldSynthesisOptions options, IWorldSynthesizer synthesizer,
        IWorldCodec? codec = null)
    {
        var (f0, sp, ap) = GenWorldParams(p, options, codec);
        return synthesizer.Synthesize(f0, sp, ap, CheapTrickFftSize(options.SampleRate), options.FramePeriod,
            options.SampleRate);
    }

    private static IWorldCodec Codec(IWorldCodec? codec)
        => codec ?? throw new InvalidOperationException(
            "WORLD の codec (IWorldCodec、たとえば WorldlineNative) が必要です");
}
