using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxSvs;

/// <summary>モデルの出力。確率モデル (MDN) なら Mu と Sigma、そうでなければ Y だけ。</summary>
public sealed record StageOutput(float[][] Y, float[][]? Sigma)
{
    public bool IsProbabilistic => Sigma != null;
}

/// <summary>onnx_export の model_config (MLPG などに使う設定)。</summary>
public sealed record ModelConfig(int[] StreamSizes, bool[] HasDynamicFeatures, int NumWindows = 1)
{
    public bool AnyDynamicFeatures => HasDynamicFeatures.Any(x => x);
}

/// <summary>
/// onnx_export が書き出した 1 段 (timelag / duration / acoustic) のモデル。
/// manifest.json から、onnx ファイル、入出力名、scaler、設定を読む。
/// </summary>
public sealed class ModelStage : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string[] _outputs;

    public string Name { get; }
    public int InDim { get; }
    public Scaler In { get; }
    public Scaler Out { get; }
    public ModelConfig Config { get; }
    public bool IsProbabilistic => _outputs.Length == 2;

    private ModelStage(string name, InferenceSession session, string[] outputs, int inDim,
        Scaler input, Scaler output, ModelConfig config)
    {
        Name = name;
        _session = session;
        _outputs = outputs;
        InDim = inDim;
        In = input;
        Out = output;
        Config = config;
    }

    /// <param name="onnxDir">onnx_export の出力フォルダ (manifest.json がある)</param>
    /// <param name="stage">timelag / duration / acoustic</param>
    public static ModelStage Load(string onnxDir, string stage)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(onnxDir, "manifest.json")));
        var entry = doc.RootElement.GetProperty("stages").GetProperty(stage);
        if (entry.TryGetProperty("error", out var error))
        {
            throw new NotSupportedException($"{stage} は書き出せていません: {error.GetString()}");
        }
        var (input, output) = ScalerLoader.Load(entry);
        if (input == null || output == null)
        {
            throw new InvalidDataException($"{stage} の scaler が manifest.json にありません");
        }
        var outputs = entry.GetProperty("outputs").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var cfg = entry.GetProperty("model_config");
        var config = new ModelConfig(
            cfg.GetProperty("stream_sizes").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            cfg.GetProperty("has_dynamic_features").EnumerateArray().Select(e => e.GetBoolean()).ToArray(),
            cfg.TryGetProperty("num_windows", out var nw) ? nw.GetInt32() : 1);
        var session = new InferenceSession(Path.Combine(onnxDir, entry.GetProperty("file").GetString()!));
        return new ModelStage(stage, session, outputs, entry.GetProperty("in_dim").GetInt32(),
            input, output, config);
    }

    /// <summary>x は (フレーム, 次元)。バッチ 1 として実行する。</summary>
    public StageOutput Run(float[][] x)
    {
        var frames = x.Length;
        var flat = new float[frames * InDim];
        for (var t = 0; t < frames; t++)
        {
            if (x[t].Length != InDim)
            {
                throw new ArgumentException($"{Name}: 入力の次元が {x[t].Length} で、モデルは {InDim} です");
            }
            x[t].CopyTo(flat, t * InDim);
        }
        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("x", new DenseTensor<float>(flat, new[] { 1, frames, InDim })),
        };
        using var results = _session.Run(inputs, _outputs);
        var arrays = results.Select(r => ToRows(r.AsTensor<float>())).ToArray();
        return new StageOutput(arrays[0], arrays.Length == 2 ? arrays[1] : null);
    }

    private static float[][] ToRows(Tensor<float> tensor)
    {
        // (1, T, D) -> (T, D)
        var frames = tensor.Dimensions[1];
        var dim = tensor.Dimensions[2];
        var flat = tensor.ToArray();
        var rows = new float[frames][];
        for (var t = 0; t < frames; t++)
        {
            rows[t] = new float[dim];
            Array.Copy(flat, t * dim, rows[t], 0, dim);
        }
        return rows;
    }

    public void Dispose() => _session.Dispose();
}
