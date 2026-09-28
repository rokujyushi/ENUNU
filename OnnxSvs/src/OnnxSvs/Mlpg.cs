namespace OnnxSvs;

/// <summary>MLPG の窓 (l, u, 係数)。列は t - l から t + u。</summary>
public readonly record struct MlpgWindow(int Left, int Right, double[] Coefficients);

/// <summary>
/// 最尤パラメータ生成 (MLPG)。nnmnkwii.paramgen.mlpg と nnsvs.multistream の multi_stream_mlpg を移したもの。
/// 静的特徴 y を、A y = b (A = Σ Wᵀ P W、b = Σ Wᵀ P μ) を解いて求める。
/// </summary>
public static class Mlpg
{
    /// <summary>nnsvs.multistream.get_windows: 静的、デルタ、デルタデルタ。</summary>
    public static MlpgWindow[] GetWindows(int numWindows)
    {
        var windows = new List<MlpgWindow> { new(0, 0, new[] { 1.0 }) };
        if (numWindows >= 2)
        {
            windows.Add(new MlpgWindow(1, 1, new[] { -0.5, 0.0, 0.5 }));
        }
        if (numWindows >= 3)
        {
            windows.Add(new MlpgWindow(1, 1, new[] { 1.0, -2.0, 1.0 }));
        }
        if (numWindows >= 4)
        {
            throw new ArgumentException($"Not supported num windows: {numWindows}");
        }
        return windows.ToArray();
    }

    /// <summary>
    /// (T, 静的次元 × 窓数) の平均と分散から、(T, 静的次元) の静的特徴を作る。
    /// 分散は (T, D) か、全フレーム共通の 1 次元 (長さ D) を渡す。
    /// </summary>
    public static double[][] Generate(double[][] mean, double[][] variance, MlpgWindow[] windows)
    {
        var T = mean.Length;
        var D = mean[0].Length;
        var numWindows = windows.Length;
        if (D % numWindows != 0)
        {
            throw new ArgumentException("次元が窓の数で割り切れません");
        }
        var staticDim = D / numWindows;
        var maxWidth = windows.Max(w => Math.Max(w.Left, w.Right));

        var y = new double[T][];
        for (var t = 0; t < T; t++)
        {
            y[t] = new double[staticDim];
        }

        var means = new double[numWindows][];
        var precisions = new double[numWindows][];
        for (var w = 0; w < numWindows; w++)
        {
            means[w] = new double[T];
            precisions[w] = new double[T];
        }

        for (var d = 0; d < staticDim; d++)
        {
            for (var w = 0; w < numWindows; w++)
            {
                for (var t = 0; t < T; t++)
                {
                    means[w][t] = mean[t][w * staticDim + d];
                    precisions[w][t] = 1.0 / variance[t][w * staticDim + d];
                }
                // 動的特徴は、両端の窓がはみ出すフレームでは精度を 0 にする
                if (w != 0)
                {
                    for (var t = 0; t < Math.Min(maxWidth, T); t++)
                    {
                        precisions[w][t] = 0;
                    }
                    for (var t = Math.Max(0, T - maxWidth); t < T; t++)
                    {
                        precisions[w][t] = 0;
                    }
                }
            }
            var solution = Solve(means, precisions, windows, T);
            for (var t = 0; t < T; t++)
            {
                y[t][d] = solution[t];
            }
        }
        return y;
    }

    /// <summary>分散が全フレーム共通 (長さ D) の場合。</summary>
    public static double[][] Generate(double[][] mean, double[] globalVariance, MlpgWindow[] windows)
        => Generate(mean, mean.Select(_ => globalVariance).ToArray(), windows);

    /// <summary>1 次元分の A y = b を作って解く。A は幅 2 * (窓の最大幅) の帯行列。</summary>
    private static double[] Solve(double[][] means, double[][] precisions, MlpgWindow[] windows, int T)
    {
        var half = 2 * windows.Max(w => Math.Max(w.Left, w.Right));
        // 下三角の帯: a[i][i - j] (0 <= i - j <= half) が A[i, j]
        var a = new double[T][];
        for (var i = 0; i < T; i++)
        {
            a[i] = new double[half + 1];
        }
        var b = new double[T];

        for (var w = 0; w < windows.Length; w++)
        {
            var win = windows[w];
            for (var t = 0; t < T; t++)
            {
                var p = precisions[w][t];
                if (p == 0)
                {
                    continue;
                }
                var pm = p * means[w][t];
                for (var k1 = 0; k1 < win.Coefficients.Length; k1++)
                {
                    var c1 = t - win.Left + k1;
                    if (c1 < 0 || c1 >= T)
                    {
                        continue;
                    }
                    b[c1] += win.Coefficients[k1] * pm;
                    for (var k2 = 0; k2 < win.Coefficients.Length; k2++)
                    {
                        var c2 = t - win.Left + k2;
                        if (c2 < 0 || c2 >= T || c2 > c1)
                        {
                            continue;
                        }
                        a[c1][c1 - c2] += win.Coefficients[k1] * p * win.Coefficients[k2];
                    }
                }
            }
        }
        return SolveBandedSymmetric(a, b, half);
    }

    /// <summary>帯行列の Cholesky 分解 (A = L Lᵀ) で A x = b を解く。</summary>
    private static double[] SolveBandedSymmetric(double[][] a, double[] b, int half)
    {
        var n = b.Length;
        // L を a に上書きする
        for (var i = 0; i < n; i++)
        {
            for (var j = Math.Max(0, i - half); j <= i; j++)
            {
                var sum = a[i][i - j];
                for (var k = Math.Max(Math.Max(0, i - half), j - half); k < j; k++)
                {
                    sum -= a[i][i - k] * a[j][j - k];
                }
                if (i == j)
                {
                    if (sum <= 0)
                    {
                        throw new InvalidOperationException("MLPG: 行列が正定値ではありません");
                    }
                    a[i][0] = Math.Sqrt(sum);
                }
                else
                {
                    a[i][i - j] = sum / a[j][0];
                }
            }
        }
        // L z = b
        var z = new double[n];
        for (var i = 0; i < n; i++)
        {
            var sum = b[i];
            for (var k = Math.Max(0, i - half); k < i; k++)
            {
                sum -= a[i][i - k] * z[k];
            }
            z[i] = sum / a[i][0];
        }
        // Lᵀ x = z
        var x = new double[n];
        for (var i = n - 1; i >= 0; i--)
        {
            var sum = z[i];
            for (var k = i + 1; k <= Math.Min(n - 1, i + half); k++)
            {
                sum -= a[k][k - i] * x[k];
            }
            x[i] = sum / a[i][0];
        }
        return x;
    }

    /// <summary>
    /// nnsvs.multistream.multi_stream_mlpg: ストリームごとに、動的特徴があれば MLPG、なければそのまま。
    /// 結果は静的特徴をストリーム順に連結したもの。
    /// </summary>
    public static double[][] MultiStream(double[][] inputs, double[][] variances, MlpgWindow[] windows,
        int[] streamSizes, bool[] hasDynamicFeatures)
    {
        var T = inputs.Length;
        if (inputs[0].Length != streamSizes.Sum())
        {
            throw new ArgumentException("You probably have specified wrong dimension params.");
        }
        var pieces = new List<double[][]>();
        var start = 0;
        for (var s = 0; s < streamSizes.Length; s++)
        {
            var size = streamSizes[s];
            var st = start;
            var x = inputs.Select(row => row[st..(st + size)]).ToArray();
            if (hasDynamicFeatures[s])
            {
                var v = variances.Select(row => row[st..(st + size)]).ToArray();
                pieces.Add(Generate(x, v, windows));
            }
            else
            {
                pieces.Add(x);
            }
            start += size;
        }
        // 連結
        var result = new double[T][];
        for (var t = 0; t < T; t++)
        {
            result[t] = pieces.SelectMany(p => p[t]).ToArray();
        }
        return result;
    }
}
