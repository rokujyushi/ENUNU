using System.Globalization;

namespace OnnxSvs;

/// <summary>
/// ラベルから言語特徴量を作る。nnmnkwii.frontend.merlin の
/// load_labels_with_phone_alignment のうち、nnsvs が使う 2 通りを移したもの。
/// - 音素単位 (subphone_features=None, add_frame_features=False): timelag / duration 用
/// - フレーム単位 (subphone_features="coarse_coding", add_frame_features=True): acoustic 用
/// </summary>
public static class LinguisticFeatures
{
    /// <summary>コーステコーディングでフレームごとに足す特徴の数 (位置 3 つ + 音素長)。</summary>
    public const int CoarseCodingSize = 4;

    private static readonly Dictionary<string, int> NoteMapping = BuildNoteMapping();

    private static Dictionary<string, int> BuildNoteMapping()
    {
        string[] names = { "C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B" };
        var map = new Dictionary<string, int>();
        for (var midi = 21; midi <= 127; midi++)
        {
            map[names[midi % 12] + (midi / 12 - 1)] = midi;
        }
        return map;
    }

    /// <summary>ラベル 1 行を 0/1 の特徴にする。</summary>
    public static float[] MatchBinary(QuestionSet qs, string label)
    {
        var vec = new float[qs.Binary.Count];
        for (var i = 0; i < vec.Length; i++)
        {
            foreach (var pattern in qs.Binary[i].Patterns)
            {
                if (pattern.IsMatch(label))
                {
                    vec[i] = 1;
                    break;
                }
            }
        }
        return vec;
    }

    /// <summary>ラベル 1 行を、音高や位置などの数値の特徴にする。</summary>
    public static float[] MatchNumeric(QuestionSet qs, string label)
    {
        var vec = new float[qs.Numeric.Count];
        for (var i = 0; i < vec.Length; i++)
        {
            var pattern = qs.Numeric[i].Pattern;
            var value = pattern.ToString().Contains("([-\\d]+)") ? -50f : -1f;
            var m = pattern.Match(label);
            if (m.Success)
            {
                var s = m.Groups[1].Value;
                if (NoteMapping.TryGetValue(s, out var midi))
                {
                    value = midi;
                }
                else if (s.StartsWith('p'))
                {
                    value = int.Parse(s[1..], CultureInfo.InvariantCulture);
                }
                else if (s.StartsWith('m'))
                {
                    value = -int.Parse(s[1..], CultureInfo.InvariantCulture);
                }
                else
                {
                    value = float.Parse(s, CultureInfo.InvariantCulture);
                }
            }
            vec[i] = value;
        }
        return vec;
    }

    /// <summary>音素ごとに 1 行の特徴 (音素数, 次元)。</summary>
    public static float[][] PhoneLevel(HtsLabel labels, QuestionSet qs)
    {
        return labels.Lines.Select(l => PhoneVector(qs, l.Context)).ToArray();
    }

    /// <summary>フレームごとに 1 行の特徴 (フレーム数, 次元 + 4)。</summary>
    public static float[][] FrameLevelCoarseCoding(
        HtsLabel labels, QuestionSet qs, long frameShift = HtsLabel.DefaultFrameShift)
    {
        var cc = CoarseCodingTable();
        var rows = new List<float[]>();
        foreach (var line in labels.Lines)
        {
            var frames = (int)(line.EndTime / frameShift - line.StartTime / frameShift);
            var phone = PhoneVector(qs, line.Context);
            var ccRows = ExtractCoarseCodingRelative(cc, frames);
            for (var i = 0; i < frames; i++)
            {
                var row = new float[phone.Length + CoarseCodingSize];
                phone.CopyTo(row, 0);
                row[phone.Length + 0] = ccRows[i, 0];
                row[phone.Length + 1] = ccRows[i, 1];
                row[phone.Length + 2] = ccRows[i, 2];
                row[phone.Length + 3] = frames;
                rows.Add(row);
            }
        }
        return rows.ToArray();
    }

    private static float[] PhoneVector(QuestionSet qs, string context)
    {
        var binary = MatchBinary(qs, context);
        var numeric = MatchNumeric(qs, context);
        var vec = new float[binary.Length + numeric.Length];
        binary.CopyTo(vec, 0);
        numeric.CopyTo(vec, binary.Length);
        return vec;
    }

    /// <summary>compute_coarse_coding_features: 3 状態のガウス分布 (σ=0.4) を 600 点で並べた表。</summary>
    internal static double[][] CoarseCodingTable(int npoints = 600)
    {
        double[] starts = { -1.5, -1.0, -0.5 };
        double[] stops = { 1.5, 2.0, 2.5 };
        double[] mus = { 0.0, 0.5, 1.0 };
        const double sigma = 0.4;
        var table = new double[3][];
        for (var k = 0; k < 3; k++)
        {
            table[k] = new double[npoints];
            var step = (stops[k] - starts[k]) / (npoints - 1);
            for (var i = 0; i < npoints; i++)
            {
                // numpy.linspace は最後の点をちょうど stop にする
                var x = i == npoints - 1 ? stops[k] : starts[k] + i * step;
                var z = (x - mus[k]) / sigma;
                table[k][i] = Math.Exp(-0.5 * z * z) / (sigma * Math.Sqrt(2 * Math.PI));
            }
        }
        return table;
    }

    /// <summary>extract_coarse_coding_features_relative</summary>
    internal static float[,] ExtractCoarseCodingRelative(double[][] cc, int duration)
    {
        var result = new float[duration, 3];
        for (var i = 0; i < duration; i++)
        {
            var rel = (int)(200 / (double)duration * i);
            result[i, 0] = (float)cc[0][300 + rel];
            result[i, 1] = (float)cc[1][200 + rel];
            result[i, 2] = (float)cc[2][100 + rel];
        }
        return result;
    }
}
