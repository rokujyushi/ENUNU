using System.Numerics;

namespace OnnxSvs;

/// <summary>scipy.signal.butter / filtfilt と同じ結果になる、ゼロ位相のローパスフィルタ (nnsvs.dsp.lowpass_filter)。</summary>
public static class Dsp
{
    /// <summary>nnsvs.dsp.lowpass_filter。信号が短すぎるときは何もしない。</summary>
    public static double[] LowpassFilter(double[] x, int fs, double cutoff = 5, int order = 5)
    {
        var (b, a) = Butter(order, cutoff / (fs / 2));
        if (x.Length <= Math.Max(a.Length, b.Length) * (order / 2 + 1))
        {
            return (double[])x.Clone();
        }
        return FiltFilt(b, a, x);
    }

    /// <summary>scipy.signal.butter(N, Wn, "lowpass") (デジタル、Wn は正規化周波数)。</summary>
    public static (double[] B, double[] A) Butter(int order, double wn)
    {
        const double fs = 2.0;
        var warped = 2 * fs * Math.Tan(Math.PI * wn / fs);
        // アナログのバターワース原型の極を、遮断周波数に合わせて広げる
        var poles = new List<Complex>();
        for (var m = -order + 1; m < order; m += 2)
        {
            poles.Add(-Complex.Exp(new Complex(0, Math.PI * m / (2.0 * order))) * warped);
        }
        var k = Math.Pow(warped, order);
        // 双一次変換
        var fs2 = 2 * fs;
        var polesZ = poles.Select(p => (fs2 + p) / (fs2 - p)).ToArray();
        var prod = Complex.One;
        foreach (var p in poles)
        {
            prod *= fs2 - p;
        }
        var kz = k * (1 / prod).Real;
        var zerosZ = Enumerable.Repeat(new Complex(-1, 0), order).ToArray();
        var b = Poly(zerosZ).Select(c => c.Real * kz).ToArray();
        var a = Poly(polesZ).Select(c => c.Real).ToArray();
        return (b, a);
    }

    private static Complex[] Poly(Complex[] roots)
    {
        var c = new List<Complex> { Complex.One };
        foreach (var r in roots)
        {
            var next = new Complex[c.Count + 1];
            for (var i = 0; i < c.Count; i++)
            {
                next[i] += c[i];
                next[i + 1] -= c[i] * r;
            }
            c = next.ToList();
        }
        return c.ToArray();
    }

    /// <summary>scipy.signal.filtfilt (padtype="odd", padlen=3*max(len(a), len(b)))。</summary>
    public static double[] FiltFilt(double[] b, double[] a, double[] x)
    {
        var n = 3 * Math.Max(a.Length, b.Length);
        if (x.Length <= n)
        {
            throw new ArgumentException("信号が短すぎます");
        }
        var ext = new double[x.Length + 2 * n];
        for (var i = 0; i < n; i++)
        {
            ext[i] = 2 * x[0] - x[n - i];
            ext[n + x.Length + i] = 2 * x[^1] - x[^(2 + i)];
        }
        Array.Copy(x, 0, ext, n, x.Length);

        var zi = LfilterZi(b, a);
        var y = Lfilter(b, a, ext, zi.Select(z => z * ext[0]).ToArray());
        Array.Reverse(y);
        y = Lfilter(b, a, y, zi.Select(z => z * y[0]).ToArray());
        Array.Reverse(y);
        return y[n..^n];
    }

    /// <summary>直接形 II 転置の IIR フィルタ (scipy.signal.lfilter)。zi は初期状態。</summary>
    public static double[] Lfilter(double[] b, double[] a, double[] x, double[] zi)
    {
        var order = Math.Max(a.Length, b.Length);
        var bb = new double[order];
        var aa = new double[order];
        for (var i = 0; i < b.Length; i++)
        {
            bb[i] = b[i] / a[0];
        }
        for (var i = 0; i < a.Length; i++)
        {
            aa[i] = a[i] / a[0];
        }
        var z = new double[order];
        Array.Copy(zi, z, order - 1);
        var y = new double[x.Length];
        for (var n = 0; n < x.Length; n++)
        {
            y[n] = bb[0] * x[n] + z[0];
            for (var i = 1; i < order; i++)
            {
                z[i - 1] = bb[i] * x[n] - aa[i] * y[n] + z[i];
            }
        }
        return y;
    }

    /// <summary>scipy.signal.lfilter_zi: ステップ入力に対する定常状態。</summary>
    public static double[] LfilterZi(double[] b, double[] a)
    {
        var n = Math.Max(a.Length, b.Length);
        var bb = new double[n];
        var aa = new double[n];
        for (var i = 0; i < b.Length; i++)
        {
            bb[i] = b[i] / a[0];
        }
        for (var i = 0; i < a.Length; i++)
        {
            aa[i] = a[i] / a[0];
        }
        var m = n - 1;
        // (I - companion(a)^T) zi = b[1:] - a[1:] * b[0]
        var mat = new double[m, m + 1];
        for (var i = 0; i < m; i++)
        {
            mat[i, i] += 1;
            // companion(a): 1 行目は -a[1:]、下の副対角に 1。転置すると 1 列目が -a[1:]、上の副対角に 1
            if (i == 0)
            {
                for (var j = 0; j < m; j++)
                {
                    mat[j, 0] -= -aa[j + 1];
                }
            }
            if (i + 1 < m)
            {
                mat[i, i + 1] -= 1;
            }
            mat[i, m] = bb[i + 1] - aa[i + 1] * bb[0];
        }
        for (var col = 0; col < m; col++)
        {
            var pivot = col;
            for (var r = col + 1; r < m; r++)
            {
                if (Math.Abs(mat[r, col]) > Math.Abs(mat[pivot, col]))
                {
                    pivot = r;
                }
            }
            for (var c = 0; c <= m; c++)
            {
                (mat[col, c], mat[pivot, c]) = (mat[pivot, c], mat[col, c]);
            }
            for (var r = 0; r < m; r++)
            {
                if (r == col)
                {
                    continue;
                }
                var f = mat[r, col] / mat[col, col];
                for (var c = col; c <= m; c++)
                {
                    mat[r, c] -= f * mat[col, c];
                }
            }
        }
        return Enumerable.Range(0, m).Select(i => mat[i, m] / mat[i, i]).ToArray();
    }
}
