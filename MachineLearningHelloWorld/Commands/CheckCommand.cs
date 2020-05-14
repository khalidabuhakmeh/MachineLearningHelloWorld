using System;
using Baseline;
using MachineLearningHelloWorld.Structures;
using Microsoft.ML;
using Oakton;

namespace MachineLearningHelloWorld
{
    public class CheckCommand : OaktonCommand<CheckInput>
    {
        public CheckCommand()
        {
            Usage("default").Arguments(x => x.Text);
            Usage("with model").Arguments(x => x.Text, x => x.ModelPath);
        }
        
        public override bool Execute(CheckInput input)
        {
            var mlContext = new MLContext(seed: 1);

            var modelPath = input.ModelPath.IsEmpty()
                ? TrainingCommand.GetModelPath(TrainingCommand.DefaultModelPath)
                : input.ModelPath.EndsWith(".zip")
                    ? input.ModelPath
                    : TrainingCommand.GetModelPath(input.ModelPath);

            var transformer = mlContext.Model.Load(modelPath, out _);
            var engine =
                mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(transformer);
            
            var example = new SentimentIssue { Text = input.Text };
            var prediction = engine.Predict(example);
            var result = prediction.Prediction ? "Toxic" : "Non Toxic"; 

            Console.WriteLine($"=============== Single Prediction  ===============");
            
            Console.WriteLine($"Text: {example.Text} |" +
                              $" Prediction: {result} sentiment | " +
                              $"Probability of being toxic: {prediction.Probability} ");
            return true;
        }
    }

    public class CheckInput
    {
        [Description("the text to be analyzed")]
        public string Text { get; set; }
        
        [FlagAlias('m')]
        [Description("path of the trained model")]
        public string ModelPath { get; set; }
    }
}