using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Xml.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
namespace BikeSharingPrediction
{
    class Program
    {
        private static string _projectRoot = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "..", "..", "..");
        private static string _dataPath = Path.Combine(_projectRoot, "Resources", "bike_sharing.csv");
        public class BikeRentalData
        {
            [LoadColumn(0)]
            public float Season { get; set; }
            [LoadColumn(1)]
            public float Month { get; set; }
            [LoadColumn(2)]
            public float Hour { get; set; }
            [LoadColumn(3)]
            public float Holiday { get; set; }
            [LoadColumn(4)]
            public float Weekday { get; set; }
            [LoadColumn(5)]
            public float WorkingDay { get; set; }
            [LoadColumn(6)]
            public float WeatherCondition { get; set; }
            [LoadColumn(7)]
            public float Temperature { get; set; }
            [LoadColumn(8)]
            public float Humidity { get; set; }
            [LoadColumn(9)]
            public float Windspeed { get; set; }
            [LoadColumn(10)]
            public bool RentalType { get; set; }
        }

        public class RentalTypePrediction
        {
            [ColumnName("PredictedLabel")]
            public bool PredictedRentalType { get; set; }
            public float Probability { get; set; }
            public float Score { get; set; }
        }
        
        static void Main(string[] args)
        {

            // 1. Create ML.NET context
            var mlContext = new MLContext(seed: 0);

            // 2. Data loading
            var loader = mlContext.Data.CreateTextLoader(new TextLoader.Options
            {
                Separators = new[] { ',' }, 
                HasHeader = true,
                Columns = new[] { 
                    new TextLoader.Column("Season", DataKind.Single, 0),
                    new TextLoader.Column("Month", DataKind.Single, 1),
                    new TextLoader.Column("Hour", DataKind.Single, 2),
                    new TextLoader.Column("Holiday", DataKind.Single, 3),
                    new TextLoader.Column("Weekday", DataKind.Single, 4),
                    new TextLoader.Column("WorkingDay", DataKind.Single, 5),
                    new TextLoader.Column("WeatherCondition", DataKind.Single, 6),
                    new TextLoader.Column("Temperature", DataKind.Single, 7),
                    new TextLoader.Column("Humidity", DataKind.Single, 8),
                    new TextLoader.Column("Windspeed", DataKind.Single, 9),
                    new TextLoader.Column("RentalType", DataKind.Boolean, 10),
                }
            });
            
            var data = loader.Load(_dataPath);

            // 3. Separation of data into training and test samples(80/20)
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // 4. Creating a data processing pipeline(adding processing of categorical and numerical attributes)
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("Label", "RentalType")
                //One-hot-encode season & weather
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("SeasonEncoded", "Season"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("WeatherEncoded", "WeatherCondition"))
                // Collecting numerical features
                .Append(mlContext.Transforms.Concatenate("NumericFeatures", "Month", "Hour", "Holiday", "Weekday",
                    "WorkingDay", "Temperature", "Humidity", "Windspeed"))
                // Normalization
                .Append(mlContext.Transforms.NormalizeMinMax("NumericFeatures"))
                // Final feature vector
                .Append(mlContext.Transforms.Concatenate("Features", "SeasonEncoded", "WeatherEncoded", "NumericFeatures"))
                .AppendCacheCheckpoint(mlContext);

            // 5. Training models and choosing the best one
            var trainers = new (string name, IEstimator<ITransformer> trainer)[]
            {
                ("FastTree", mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label", featureColumnName: "Features")),
                ("LightGBM", mlContext.BinaryClassification.Trainers.LightGbm(
                    labelColumnName: "Label", featureColumnName: "Features")),
                ("LogisticRegression", mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    labelColumnName: "Label", featureColumnName: "Features")),
            };

            BinaryClassificationMetrics? bestMetrics = null;
            ITransformer? bestModel = null;
            string bestName = string.Empty;

            // 6. Assessing the quality of the model
            foreach (var (name, trainer) in trainers)
            {
                Console.WriteLine($"Trainig {name}...");
                var model = dataProcessPipeline.Append(trainer).Fit(split.TrainSet);
                var predictions = model.Transform(split.TestSet);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

                Console.WriteLine($"{name}:\tAUC = {metrics.AreaUnderRocCurve:P2}\tF1 = {metrics.F1Score:P2}");

                if (bestMetrics == null || metrics.F1Score > bestMetrics.F1Score) 
                {
                    bestMetrics = metrics;
                    bestModel = model;
                    bestName = name;
                }
            }
            
            if (bestMetrics != null) 
            { 
                Console.WriteLine($"\nBest model: {bestModel} (AUC = {bestMetrics.AreaUnderRocCurve:P2}, F1 ={bestMetrics.F1Score:P2})\n");
            }

            // 7. Fulfilment of predictions
            var predictor = mlContext.Model.CreatePredictionEngine<BikeRentalData, RentalTypePrediction>(bestModel);
            var sample = new BikeRentalData
            {
                Season = 3,
                Month = 7,
                Hour = 9,
                Holiday = 0,
                Weekday = 4,
                WorkingDay = 1,
                WeatherCondition = 1,
                Temperature = 25,
                Humidity = 60,
                Windspeed = 15
            };

            var result = predictor.Predict(sample);
            Console.WriteLine($"Prediction: {(result.PredictedRentalType ? "Long-Term": "Short-Term")} (probability {result.Probability:P1})\n");

            //8. Save model
            var modelPath = Path.Combine(_projectRoot, "Models", "BikeRentalModel.zip");
            mlContext.Model.Save(bestModel, data.Schema, modelPath);
            Console.WriteLine($"Saved to {modelPath}");
            Console.ReadKey();
        }
    }
}