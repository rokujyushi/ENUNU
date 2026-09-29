namespace OnnxSvs;

/// <summary>nnsvs.pitch の gen_sine_vibrato / nonzero_segments。</summary>
public static class Vibrato
{
    /// <summary>f0 &gt; 0 が続く区間 (開始, 終了)。終了は含まない。最後まで続く区間の終了は最終フレームの位置 (nnsvs と同じ)。</summary>
    public static List<(int Start, int End)> NonzeroSegments(double[] f0)
    {
        var segments = new List<(int, int)>();
        var started = false;
        var s = 0;
        for (var i = 0; i < f0.Length; i++)
        {
            if (f0[i] > 0 && !started)
            {
                started = true;
                s = i;
            }
            else if (started && f0[i] <= 0)
            {
                started = false;
                segments.Add((s, i));
            }
        }
        if (started && f0[^1] > 0)
        {
            segments.Add((s, f0.Length - 1));
        }
        return segments;
    }

    /// <summary>正弦波のビブラートを f0 (Hz) に足す。mA は振幅 (cent)、mF は周波数 (Hz)、sr は f0 の系列の周波数。</summary>
    public static double[] GenSineVibrato(double[] f0, int sr, double[] mA, double[] mF, double scale = 1.0)
    {
        var gen = (double[])f0.Clone();
        var voicedEnds = NonzeroSegments(f0).Select(seg => seg.End).ToArray();
        foreach (var (s, e) in NonzeroSegments(mA))
        {
            for (var i = 0; i < e - s; i++)
            {
                // ビブラートの速さは [3, 8] Hz、深さは [30, 150] cent に制限する
                var freq = Math.Clamp(mF[s + i], 3, 8);
                var extent = Math.Clamp(mA[s + i], 30, 150);
                var cent = scale * extent * Math.Sin(2 * Math.PI / sr * freq * i);
                gen[s + i] = f0[s + i] * Math.Exp(cent * Math.Log(2) / 1200);
            }
            // ビブラートの終わりで不連続にならないように、その先の有声区間の終わりまでをなめらかにする
            var next = voicedEnds.Where(v => v > e).ToArray();
            if (next.Length > 0)
            {
                var voicedEnd = next[0];
                var smooth = Dsp.LowpassFilter(gen[s..voicedEnd], sr, 12);
                Array.Copy(smooth, 0, gen, s, smooth.Length);
            }
        }
        return gen;
    }
}
