using System.Runtime.InteropServices;

namespace OnnxSvs;

/// <summary>
/// OpenUtau の Worldline (ネイティブライブラリ "worldline") を呼ぶ。宣言は OpenUtau の
/// OpenUtau.Core/Render/Worldline.cs (MIT License, Copyright (c) 2014 StAkira) の DecodeMgc / DecodeBap /
/// WorldSynthesis と同じ。ライブラリは OpenUtau に同梱のものを、実行ファイルのそばに置く。
/// </summary>
public sealed class WorldlineNative : IWorldCodec, IWorldSynthesizer
{
    [DllImport("worldline", CallingConvention = CallingConvention.Cdecl)]
    private static extern void DecodeMgc(int f0Length, double[] mgc, int mgcSize, int fftSize, int fs,
        double[] spectrogram);

    [DllImport("worldline", CallingConvention = CallingConvention.Cdecl)]
    private static extern void DecodeBap(int f0Length, double[] bap, int fftSize, int fs, double[] aperiodicity);

    [DllImport("worldline", CallingConvention = CallingConvention.Cdecl)]
    private static extern int WorldSynthesisSampleCount(int f0Length, double framePeriod, int fs);

    [DllImport("worldline", CallingConvention = CallingConvention.Cdecl)]
    private static extern int WorldSynthesis(
        double[] f0, int f0Length,
        double[] mgcOrSp, bool isMgc, int mgcSize,
        double[] bapOrAp, bool isBap, int fftSize,
        double framePeriod, int fs, double[] y,
        double[] gender, double[] tension,
        double[] breathiness, double[] voicing);

    public double[][] DecodeSpectralEnvelope(double[][] mgc, int fftSize, int fs)
    {
        var frames = mgc.Length;
        var spSize = fftSize / 2 + 1;
        var data = new double[frames * spSize];
        DecodeMgc(frames, Flatten(mgc), mgc[0].Length, fftSize, fs, data);
        return Unflatten(data, frames, spSize);
    }

    public double[][] DecodeAperiodicity(double[][] bap, int fftSize, int fs)
    {
        var frames = bap.Length;
        var apSize = fftSize / 2 + 1;
        var data = new double[frames * apSize];
        DecodeBap(frames, Flatten(bap), fftSize, fs, data);
        return Unflatten(data, frames, apSize);
    }

    public double[] Synthesize(double[] f0, double[][] sp, double[][] ap, int fftSize, double framePeriod, int fs)
    {
        var frames = f0.Length;
        var y = new double[WorldSynthesisSampleCount(frames, framePeriod, fs)];
        // gender / tension / breathiness / voicing はどれも変えない値 (0.5、0.5、0.5、1.0)
        WorldSynthesis(f0, frames, Flatten(sp), false, sp[0].Length, Flatten(ap), false, fftSize,
            framePeriod, fs, y,
            Enumerable.Repeat(0.5, frames).ToArray(), Enumerable.Repeat(0.5, frames).ToArray(),
            Enumerable.Repeat(0.5, frames).ToArray(), Enumerable.Repeat(1.0, frames).ToArray());
        return y;
    }

    private static double[] Flatten(double[][] rows)
    {
        var width = rows[0].Length;
        var flat = new double[rows.Length * width];
        for (var t = 0; t < rows.Length; t++)
        {
            Array.Copy(rows[t], 0, flat, t * width, width);
        }
        return flat;
    }

    private static double[][] Unflatten(double[] data, int frames, int width)
        => Enumerable.Range(0, frames).Select(t => data[(t * width)..((t + 1) * width)]).ToArray();
}
