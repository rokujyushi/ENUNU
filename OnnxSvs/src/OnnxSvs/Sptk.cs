namespace OnnxSvs;

/// <summary>pysptk (SPTK) の mcepalpha / freqt / mc2sp。メルケプストラムからスペクトルへの変換に使う。</summary>
public static class Sptk
{
    /// <summary>pysptk.util.mcepalpha: サンプリング周波数に合う周波数ワーピング係数。</summary>
    public static double McepAlpha(int fs, double start = 0.0, double stop = 1.0, double step = 0.001,
        int numPoints = 1000)
    {
        // メル尺度とワーピング後の周波数を [0, 1] にそろえて、二乗平均の距離が最小の alpha を選ぶ
        var melStep = fs / 2.0 / numPoints;
        var mel = new double[numPoints];
        for (var i = 0; i < numPoints; i++)
        {
            mel[i] = 1000.0 / Math.Log(2) * Math.Log(1 + melStep * i / 1000.0);
        }
        for (var i = 0; i < numPoints; i++)
        {
            mel[i] /= mel[^1];
        }
        var count = (int)Math.Ceiling((stop - start) / step);
        var best = start;
        var bestDistance = double.PositiveInfinity;
        for (var k = 0; k < count; k++)
        {
            var alpha = start + k * step;
            var warp = WarpingVector(alpha, numPoints);
            var distance = 0.0;
            for (var i = 0; i < numPoints; i++)
            {
                var d = mel[i] - warp[i];
                distance += Math.Abs(d * d);
            }
            distance /= numPoints;
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = alpha;
            }
        }
        return best;
    }

    private static double[] WarpingVector(double alpha, int length)
    {
        var step = Math.PI / length;
        var warp = new double[length];
        for (var i = 0; i < length; i++)
        {
            var omega = step * i;
            var num = (1 - alpha * alpha) * Math.Sin(omega);
            var den = (1 + alpha * alpha) * Math.Cos(omega) - 2 * alpha;
            var w = Math.Atan(num / den);
            warp[i] = w < 0 ? w + Math.PI : w;
        }
        for (var i = 0; i < length; i++)
        {
            warp[i] /= warp[^1];
        }
        return warp;
    }

    /// <summary>SPTK の freqt: ケプストラム c1 (次数 m1) を、周波数ワーピング alpha をかけた次数 m2 のケプストラムにする。</summary>
    public static double[] Freqt(double[] c1, int m2, double alpha)
    {
        var m1 = c1.Length - 1;
        var b = 1 - alpha * alpha;
        var d = new double[m2 + 1];
        var g = new double[m2 + 1];
        for (var i = -m1; i <= 0; i++)
        {
            d[0] = g[0];
            g[0] = c1[-i] + alpha * d[0];
            if (m2 >= 1)
            {
                d[1] = g[1];
                g[1] = b * d[0] + alpha * d[1];
            }
            for (var j = 2; j <= m2; j++)
            {
                d[j] = g[j];
                g[j] = d[j - 1] + alpha * (d[j] - g[j - 1]);
            }
        }
        return g;
    }

    /// <summary>pysptk.mc2sp: メルケプストラム 1 フレームをパワースペクトル (fftLen/2 + 1 点) にする。</summary>
    public static double[] Mc2Sp(double[] mc, double alpha, int fftLen)
    {
        var c = Freqt(mc, fftLen / 2, -alpha);
        c[0] *= 2.0;
        // 偶対称のケプストラム (symc[-i] = c[i]) の DFT の実部。対称なので虚部は 0
        var symc = new double[fftLen];
        symc[0] = c[0];
        for (var i = 1; i < c.Length; i++)
        {
            symc[i] = c[i];
            symc[fftLen - i] = c[i];
        }
        var real = RealPartOfDft(symc);
        var sp = new double[fftLen / 2 + 1];
        for (var k = 0; k < sp.Length; k++)
        {
            sp[k] = Math.Exp(real[k]);
        }
        return sp;
    }

    /// <summary>実数列の DFT の実部 (先頭 n/2 + 1 点)。n が 2 のべき乗なら FFT、そうでなければ素直な DFT。</summary>
    internal static double[] RealPartOfDft(double[] x)
    {
        var n = x.Length;
        var half = n / 2 + 1;
        var result = new double[half];
        if ((n & (n - 1)) != 0)
        {
            for (var k = 0; k < half; k++)
            {
                var sum = 0.0;
                for (var i = 0; i < n; i++)
                {
                    sum += x[i] * Math.Cos(2 * Math.PI * k * i / n);
                }
                result[k] = sum;
            }
            return result;
        }
        var re = (double[])x.Clone();
        var im = new double[n];
        // ビット反転の並べ替え
        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }
            j ^= bit;
            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
            }
        }
        for (var len = 2; len <= n; len <<= 1)
        {
            var angle = -2 * Math.PI / len;
            var wr = Math.Cos(angle);
            var wi = Math.Sin(angle);
            for (var i = 0; i < n; i += len)
            {
                double cr = 1, ci = 0;
                for (var k = 0; k < len / 2; k++)
                {
                    var a = i + k;
                    var b = i + k + len / 2;
                    var tr = re[b] * cr - im[b] * ci;
                    var ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    (cr, ci) = (cr * wr - ci * wi, cr * wi + ci * wr);
                }
            }
        }
        Array.Copy(re, result, half);
        return result;
    }
}
