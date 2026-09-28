using System.Text.Json;

namespace OnnxSvs;

/// <summary>特徴量の正規化。nnsvs.util の MinMaxScaler / StandardScaler と同じ式。</summary>
public abstract class Scaler
{
    public abstract float Transform(float x, int dim);
    public abstract float InverseTransform(float x, int dim);

    /// <summary>行ごとに変換した新しい配列を返す。</summary>
    public float[][] Transform(float[][] rows)
        => rows.Select(row => row.Select((x, d) => Transform(x, d)).ToArray()).ToArray();

    public float[][] InverseTransform(float[][] rows)
        => rows.Select(row => row.Select((x, d) => InverseTransform(x, d)).ToArray()).ToArray();
}

/// <summary>x * scale + min</summary>
public sealed class MinMaxScaler : Scaler
{
    private readonly float[] _min;
    private readonly float[] _scale;

    public MinMaxScaler(float[] min, float[] scale)
    {
        _min = min;
        _scale = scale;
    }

    public override float Transform(float x, int dim) => _scale[dim] * x + _min[dim];
    public override float InverseTransform(float x, int dim) => (x - _min[dim]) / _scale[dim];
}

/// <summary>(x - mean) / scale</summary>
public sealed class StandardScaler : Scaler
{
    private readonly float[] _mean;
    private readonly float[] _scale;

    public StandardScaler(float[] mean, float[] scale)
    {
        _mean = mean;
        _scale = scale;
    }

    public float[] Mean => _mean;
    public float[] Scale => _scale;

    public override float Transform(float x, int dim) => (x - _mean[dim]) / _scale[dim];
    public override float InverseTransform(float x, int dim) => x * _scale[dim] + _mean[dim];
}

/// <summary>onnx_export の manifest.json から scaler を読む。</summary>
public static class ScalerLoader
{
    public static (Scaler? In, Scaler? Out) Load(string manifestPath, string stage)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
        var scalers = doc.RootElement.GetProperty("stages").GetProperty(stage).GetProperty("scalers");
        Scaler? input = null, output = null;
        if (scalers.TryGetProperty("in", out var inEl))
        {
            input = new MinMaxScaler(Floats(inEl, "min"), Floats(inEl, "scale"));
        }
        if (scalers.TryGetProperty("out", out var outEl))
        {
            output = new StandardScaler(Floats(outEl, "mean"), Floats(outEl, "scale"));
        }
        return (input, output);
    }

    private static float[] Floats(JsonElement parent, string name)
        => parent.GetProperty(name).EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
}
