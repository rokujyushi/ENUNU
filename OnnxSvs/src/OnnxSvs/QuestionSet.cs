using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace OnnxSvs;

/// <summary>
/// HTS の質問ファイル (.hed) から作った、ラベルを特徴量にするための正規表現の集まり。
/// nnmnkwii.io.hts.load_question_set と同じ結果になる (QS が 0/1 の特徴、CQS が数値の特徴)。
/// </summary>
public sealed class QuestionSet
{
    /// <summary>QS: 質問ごとの正規表現のリスト。どれかに一致すれば 1。</summary>
    public IReadOnlyList<(string Name, Regex[] Patterns)> Binary { get; }

    /// <summary>CQS: 質問ごとの正規表現。最初のグループが数値。</summary>
    public IReadOnlyList<(string Name, Regex Pattern)> Numeric { get; }

    private QuestionSet(
        List<(string, Regex[])> binary, List<(string, Regex)> numeric)
    {
        Binary = binary;
        Numeric = numeric;
    }

    public int Dimension => Binary.Count + Numeric.Count;

    public static QuestionSet Load(string path, bool appendHatForLL = true, bool convertSvsPattern = true)
        => Parse(File.ReadAllLines(path), appendHatForLL, convertSvsPattern);

    public static QuestionSet Parse(
        IEnumerable<string> lines, bool appendHatForLL = true, bool convertSvsPattern = true)
    {
        var binary = new List<(string, Regex[])>();
        var numeric = new List<(string, Regex)>();
        foreach (var raw in lines)
        {
            var line = raw.Replace("\n", "").Replace("\r", "");
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }
            var words = line.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var name = words[1].Replace("\"", "").Replace("'", "");
            var body = line.Split('{')[1].Split('}')[0].Trim();
            var questions = body.Split(',');
            // Python の line.split(" ")[1] と同じ (LL- の判定に使う)
            var questionKey = line.Split(' ')[1];

            if (words[0] == "CQS")
            {
                if (questions.Length != 1)
                {
                    throw new FormatException($"CQS must have one question: {line}");
                }
                var pattern = Wildcards2Regex(questions[0], convertNumberPattern: true,
                    convertSvsPattern: convertSvsPattern);
                numeric.Add((name, new Regex(pattern, RegexOptions.CultureInvariant)));
            }
            else if (words[0] == "QS")
            {
                var patterns = new List<Regex>();
                foreach (var q in questions)
                {
                    var pattern = Wildcards2Regex(q, convertNumberPattern: false,
                        convertSvsPattern: convertSvsPattern);
                    if (appendHatForLL && questionKey.Contains("LL-") && pattern[0] != '^')
                    {
                        pattern = "^" + pattern;
                    }
                    patterns.Add(new Regex(pattern, RegexOptions.CultureInvariant));
                }
                binary.Add((name, patterns.ToArray()));
            }
            else
            {
                throw new FormatException("Not supported question format");
            }
        }
        return new QuestionSet(binary, numeric);
    }

    // Python 3.7 以降の re.escape と同じ文字だけをエスケープする
    private const string PythonEscaped = "()[]{}?*+-|^$\\.&~# \t\n\r\v\f";

    private static string PythonReEscape(string s)
    {
        var sb = new StringBuilder();
        foreach (var c in s)
        {
            if (PythonEscaped.Contains(c))
            {
                sb.Append('\\');
            }
            sb.Append(c);
        }
        return sb.ToString();
    }

    /// <summary>nnmnkwii.io.hts.wildcards2regex と同じ変換。</summary>
    public static string Wildcards2Regex(
        string question, bool convertNumberPattern = false, bool convertSvsPattern = true)
    {
        var prefix = "";
        var postfix = "";
        if (question.Contains('*'))
        {
            if (!question.StartsWith('*'))
            {
                prefix = "\\A";
            }
            if (!question.EndsWith('*'))
            {
                postfix = "\\z";
            }
        }
        question = question.Trim('*');
        question = PythonReEscape(question);
        question = question.Replace("\\*", ".*");
        question = prefix + question + postfix;

        if (convertNumberPattern)
        {
            question = question.Replace("\\(\\\\d\\+\\)", "(\\d+)");
            question = question.Replace("\\(\\[\\-\\\\d\\]\\+\\)", "([-\\d]+)");
            question = question.Replace("\\(\\[\\\\d\\\\\\.\\]\\+\\)", "([\\d\\.]+)");
        }
        if (convertSvsPattern)
        {
            question = question.Replace(
                "\\(\\[A\\-Z\\]\\[b\\]\\?\\[0\\-9\\]\\+\\)", "([A-Z][b]?[0-9]+)");
            question = question.Replace("\\(\\\\NOTE\\)", "([A-Z][b]?[0-9]+)");
            question = question.Replace("\\(\\[pm\\]\\\\d\\+\\)", "([pm]\\d+)");
        }
        return question;
    }
}
