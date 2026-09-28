namespace OnnxSvs;

/// <summary>
/// モデルに渡す前の入力の加工。nnsvs.gen の predict_timelag / predict_duration と同じ手順。
/// </summary>
public static class Conditioning
{
    /// <summary>nnsvs.io.hts.get_pitch_indices: /D /E /F で始まる数値特徴 (音高) の列番号。</summary>
    public static int[] GetPitchIndices(QuestionSet qs)
    {
        bool IsPitch(int idx)
        {
            var p = qs.Numeric[idx].Pattern.ToString();
            return p.StartsWith("/D") || p.StartsWith("/E") || p.StartsWith("/F");
        }
        if (qs.Numeric.Count == 0 || !IsPitch(0))
        {
            throw new InvalidDataException("最初の数値特徴 (CQS) が音高 (/D /E /F) ではありません");
        }
        var start = qs.Binary.Count;
        var indices = new List<int> { start };
        for (var idx = 1; idx < qs.Numeric.Count && IsPitch(idx); idx++)
        {
            indices.Add(start + idx);
        }
        return indices.ToArray();
    }

    /// <summary>nnsvs.io.hts.get_note_indices: 開始時刻が変わる行 (音符の先頭の音素) の番号。</summary>
    public static List<int> GetNoteIndices(HtsLabel labels)
    {
        var indices = new List<int> { 0 };
        var last = labels.Lines[0].StartTime;
        for (var i = 1; i < labels.Count; i++)
        {
            if (labels.Lines[i].StartTime != last)
            {
                indices.Add(i);
                last = labels.Lines[i].StartTime;
            }
        }
        return indices;
    }

    /// <summary>
    /// MIDI ノート番号の列を log-f0 にして、0 (音高なし) の区間を線形補間で埋める。
    /// _midi_to_hz(log_f0=True) と nnmnkwii.preprocessing.f0.interp1d(kind="slinear") と同じ。
    /// </summary>
    public static void ConditionPitchColumn(float[][] features, int column, double f0ShiftInCent = 0)
    {
        var n = features.Length;
        var f0 = new double[n];
        for (var i = 0; i < n; i++)
        {
            var midi = features[i][column];
            if (midi > 0)
            {
                // librosa.midi_to_hz と、その log
                f0[i] = Math.Log(440.0 * Math.Pow(2.0, (midi - 69.0) / 12.0));
            }
        }
        var filled = Interp1d(f0);
        // nnsvs は、補間したあとの log-f0 に cent 分のずれを足す
        var offset = f0ShiftInCent != 0 ? f0ShiftInCent * Math.Log(2) / 1200 : 0;
        for (var i = 0; i < n; i++)
        {
            features[i][column] = (float)(filled[i] + offset);
        }
    }

    /// <summary>nnmnkwii の interp1d。両端は最初/最後の有声値で埋め、間は線形補間。</summary>
    public static double[] Interp1d(double[] f0)
    {
        var f = (double[])f0.Clone();
        var n = f.Length;
        var nonzero = Enumerable.Range(0, n).Where(i => f[i] > 0).ToList();
        if (nonzero.Count == 0)
        {
            return f;
        }
        f[0] = f[nonzero[0]];
        f[n - 1] = f[nonzero[^1]];
        var knots = Enumerable.Range(0, n).Where(i => f[i] > 0).ToList();
        var k = 0;
        for (var i = 0; i < n; i++)
        {
            if (f[i] > 0)
            {
                continue;
            }
            while (knots[k + 1] < i)
            {
                k++;
            }
            double x0 = knots[k], x1 = knots[k + 1];
            f[i] = f[knots[k]] + (f[knots[k + 1]] - f[knots[k]]) * (i - x0) / (x1 - x0);
        }
        return f;
    }

    /// <summary>
    /// 入力の言語特徴量を、音高の加工、scaler、(必要なら) クリップの順に整えてモデルに渡せる形にする。
    /// features は書き換える。
    /// </summary>
    public static float[][] Prepare(float[][] features, ModelStage stage, int[] pitchIndices,
        bool logF0Conditioning, bool forceClipInputFeatures, double f0ShiftInCent = 0)
    {
        if (logF0Conditioning)
        {
            foreach (var idx in pitchIndices)
            {
                ConditionPitchColumn(features, idx, f0ShiftInCent);
            }
        }
        var x = stage.In.Transform(features);
        if (forceClipInputFeatures && stage.In is MinMaxScaler)
        {
            var isPitch = new HashSet<int>(pitchIndices);
            foreach (var row in x)
            {
                for (var d = 0; d < row.Length; d++)
                {
                    if (!isPitch.Contains(d))
                    {
                        row[d] = (float)Math.Clamp(row[d], MinMaxScaler.FeatureRangeMin, MinMaxScaler.FeatureRangeMax);
                    }
                }
            }
        }
        return x;
    }
}
