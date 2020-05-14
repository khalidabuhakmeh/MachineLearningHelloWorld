using Microsoft.ML.Data;

namespace MachineLearningHelloWorld.Structures
{
    public class SentimentIssue
    {
        [LoadColumn(0)]
        public bool Label { get; set; }
        [LoadColumn(2)]
        public string Text { get; set; }
    }
}