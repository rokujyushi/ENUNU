using System.Text.Json;

namespace OnnxSvs;

/// <summary>
/// 特徴量の正規化。nnsvs.util の MinMaxScaler / StandardScaler と同じ式。
/// nnsvs は scaler の値を float64 で持つので、計算は double で行う。
/// </summary>
public abstract class Scaler
{
    public abstract double Transform(double x, int dim);
    public abstract double InverseTransform(double x, int dim);

    /// <summary>行ごとに変換する。結果は float (モデルに渡す値)。</summary>
    public float[][] Transform(float[][] rows)
        => rows.Select(row => row.Select((x, d) => (float)Transform(x, d)).ToArray()).ToArray();

    /// <summary>行ごとに逆変換する。結果は double のまま返す。</summary>
    public double[][] InverseTransform(float[][] rows)
        => rows.Select(row => row.Select((x, d) => InverseTransform(x, d)).ToArray()).ToArray();
}

/// <summary>x * scale + min。feature_range は nnsvs の既定の (0, 1)。</summary>
public sealed class MinMaxScaler : Scaler
{
    public const double FeatureRangeMin = 0;
    public const double FeatureRangeMax = 1;

    public double[] Min { get; }
    public double[] Scale { get; }

    public MinMaxScaler(double[] min, double[] scale)
    {
        Min = min;
        Scale = scale;
    }

    public override double Transform(double x, int dim) => Scale[dim] * x + Min[dim];
    public override double InverseTransform(double x, int dim) => (x - Min[dim]) / Scale[dim];
}

/// <summary>(x - mean) / scale。Var は MDN の分散の戻しに使う。</summary>
public sealed class StandardScaler : Scaler
{
    public double[] Mean { get; }
    public double[] Var { get; }
    public double[] Scale { get; }

    public StandardScaler(double[] mean, double[] var, double[] scale)
    {
        Mean = mean;
        Var = var;
        Scale = scale;
    }

    public override double Transform(double x, int dim) => (x - Mean[dim]) / Scale[dim];
    public override double InverseTransform(double x, int dim) => x * Scale[dim] + Mean[dim];
}

/// <summary>onnx_export の manifest.json から scaler を読む。</summary>
public static class ScalerLoader
{
    public static (Scaler? In, Scaler? Out) Load(JsonElement stageEntry)
    {
        var scalers = stageEntry.GetProperty("scalers");
        Scaler? input = null, output = null;
        if (scalers.TryGetProperty("in", out var inEl))
        {
            input = new MinMaxScaler(Doubles(inEl, "min"), Doubles(inEl, "scale"));
        }
        if (scalers.TryGetProperty("out", out var outEl))
        {
            output = new StandardScaler(Doubles(outEl, "mean"), Doubles(outEl, "var"), Doubles(outEl, "scale"));
        }
        return (input, output);
    }

    public static (Scaler? In, Scaler? Out) Load(string manifestPath, string stage)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        return Load(doc.RootElement.GetProperty("stages").GetProperty(stage));
    }

    private static double[] Doubles(JsonElement parent, string name)
        => parent.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
}
