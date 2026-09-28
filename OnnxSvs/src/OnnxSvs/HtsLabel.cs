using System.Globalization;

namespace OnnxSvs;

/// <summary>HTS のフルコンテキストラベル 1 行。時刻は 100ns 単位。</summary>
public readonly record struct HtsLabelLine(long StartTime, long EndTime, string Context);

/// <summary>nnmnkwii.io.hts.HTSLabelFile と同じ読み方をするラベル。</summary>
public sealed class HtsLabel
{
    /// <summary>1 フレーム (5 ms) の長さ。100ns 単位。</summary>
    public const long DefaultFrameShift = 50000;

    public IReadOnlyList<HtsLabelLine> Lines { get; }

    public HtsLabel(IEnumerable<HtsLabelLine> lines)
    {
        Lines = lines.ToList();
    }

    public int Count => Lines.Count;

    public static HtsLabel Load(string path) => Parse(File.ReadAllLines(path));

    /// <summary>
    /// 「開始 終了 コンテキスト」の 3 列、またはコンテキストだけの 1 列を読む。
    /// 時刻に小数点があれば秒単位、なければ 100ns 単位。
    /// </summary>
    public static HtsLabel Parse(IEnumerable<string> lines)
    {
        var result = new List<HtsLabelLine>();
        var isSecFormat = false;
        foreach (var line in lines)
        {
            if (line.Length == 0 || line[0] == '#')
            {
                continue;
            }
            var cols = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            long start, end;
            string context;
            if (cols.Length == 3)
            {
                context = cols[2];
                if (cols[0].Contains('.') || cols[1].Contains('.'))
                {
                    isSecFormat = true;
                }
                if (isSecFormat)
                {
                    start = (long)(1e7 * double.Parse(cols[0], CultureInfo.InvariantCulture));
                    end = (long)(1e7 * double.Parse(cols[1], CultureInfo.InvariantCulture));
                }
                else
                {
                    start = long.Parse(cols[0], CultureInfo.InvariantCulture);
                    end = long.Parse(cols[1], CultureInfo.InvariantCulture);
                }
            }
            else if (cols.Length == 1)
            {
                start = -1;
                end = -1;
                context = cols[0];
            }
            else
            {
                throw new FormatException($"Not supported label line: {line}");
            }
            result.Add(new HtsLabelLine(start, end, context));
        }
        if (result.Count == 0)
        {
            throw new FormatException("Empty label");
        }
        return new HtsLabel(result);
    }

    /// <summary>
    /// 時刻をフレーム (既定 5 ms) の整数倍に丸めた新しいラベルを返す。
    /// nnmnkwii の round_ と同じく、ちょうど半分は偶数側に丸める。nnsvs は特徴量を作る前に必ずこれを行う。
    /// </summary>
    public HtsLabel Rounded(long frameShift = DefaultFrameShift)
    {
        long Round(long t) => (long)Math.Round((double)t / frameShift, MidpointRounding.ToEven) * frameShift;
        return new HtsLabel(Lines.Select(l => l with { StartTime = Round(l.StartTime), EndTime = Round(l.EndTime) }));
    }

    public long NumFrames(long frameShift = DefaultFrameShift) => Lines[^1].EndTime / frameShift;
}
