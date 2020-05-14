using System;
using System.IO;
using Common;
using MachineLearningHelloWorld.Structures;
using Microsoft.ML;
using Oakton;

namespace MachineLearningHelloWorld
{
    public class TrainingInput
    {
        [FlagAlias('i')]
        [Description("training dataset path", Name = "data")]
        public string DatasetPath { get; set; }
        
        [FlagAlias('o')]
        [Description("training model output (zip file)", Name = "output")]
        public string OutputPath { get; set; }
    }
    
    public class TrainingCommand : OaktonCommand<TrainingInput>
    {
        public const string DefaultModelPath = "./";

        public static string GetModelPath(string path)
        {
            return Path.Combine(path, "model.zip");
        }

        public TrainingCommand()
        {
            Usage("Default output").Arguments(x => x.DatasetPath);
            Usage("Override output").Arguments(x => x.DatasetPath, x => x.OutputPath);
        }
        
        public override bool Execute(TrainingInput input)
        {
            if (string.IsNullOrEmpty(input.DatasetPath))
            {
                ConsoleHelper.ConsoleWriteException("a dataset is required to train the model.");
                return false;
            }

            var outputPath = GetModelPath(
                string.IsNullOrEmpty(input.OutputPath)
                ? DefaultModelPath
                : input.OutputPath
            );
            
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            var dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(input.DatasetPath, hasHeader: true);

            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainingData = trainTestSplit.TrainSet;
            var testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));

            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // STEP 4: Train the model fitting to the DataSet
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            mlContext.Model.Save(trainedModel, trainingData.Schema, outputPath);

            Console.WriteLine("The model is saved to {0}", outputPath);

            return true;
        }
    }
}