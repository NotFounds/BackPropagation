using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using NeuralNetwork;
using NeuralNetwork.BackPropagation;

namespace NotFounds
{
    class Program
    {
        static void Main(string[] args)
        {
            var sw = new Stopwatch();

            const int    N         = 10000;
            const double Rate      = 0.05;
            const bool UseParallel = true;
            //const string TrainingDataFile = @"";
            //const string TestDataFile     = @"";

            //const int TrainingDataNum = 50000;
            //const int TestDataNum = 10000;

            const int InputSize  = 2;
            const int HiddenSize = 2;
            const int OutputSize = 1;

            // Random weight/layer
            var inputWeight  = 0.5 * Matrix.Random(HiddenSize, InputSize); //new Matrix(new double[,] { { 0.1, 0.3 }, { -0.2, 0.4 }, { 0.2, -0.3} });
            var outputWeight = 0.5 * Matrix.Random(OutputSize, HiddenSize); //new Matrix(new double[,] { { 0.5, 0.2, 0.3 }, { -0.4, 0.1, 0.2 } });
            var hiddenLayer = Matrix.Random(HiddenSize, 1); //new Matrix(new double[,] { { 0.2 }, { -0.3 }, { 0.1} });
            var outputLayer = Matrix.Random(OutputSize, 1); //new Matrix(new double[,] { { 0.4 }, { 0.2} });

            var BP = new BackPropagation(inputWeight, outputWeight, hiddenLayer, outputLayer,
                hiddenLogisticFunc: LogisticFunctions.Sigmoid,
                outputLogisticFunc: LogisticFunctions.Sigmoid,
                lossFunc: LossFunctions.MSE,
                learnRate: Rate
            );

            //Console.WriteLine($"Initial State");
            //BP.Print(4);
            //Console.WriteLine();

            // Input Training Data
            var TrainingData = new List<Pair<Matrix, Matrix>>();
            /*
            using (var sr = new StreamReader(TrainingDataFile))
            {
                for (int i = 0; i < TrainingDataNum; ++i)
                {
                    var data = sr.ReadLine().Split(',');
                    var line = new List<double>();
                    foreach (var d in data)
                        line.Add(double.Parse(d));
                    var x = new Matrix(InputSize, 1);
                    var y = new Matrix(OutputSize, 1);

                    for (int j = 0; j < InputSize; ++j)
                    {
                        x[j, 0] = line[j] / 255.0;
                    }
                    for (int j = 0; j < OutputSize; ++j)
                    {
                        y[j, 0] = line[InputSize + j];
                    }

                    TrainingData.Add(new Pair<Matrix, Matrix>(x, y));
                }
            }
            */
            TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));
            TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 1), MakeMatrix(1)));
            TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(1, 0), MakeMatrix(1)));
            TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));

            // Input Test Data
            var TestData = new List<Pair<Matrix, Matrix>>();
            /*
            using (var sr = new StreamReader(TestDataFile))
            {
                for (int i = 0; i < TestDataNum; ++i)
                {
                    var line = sr.ReadLine().Split(',').Select(s => double.Parse(s)).ToArray();
                    var x = new Matrix(InputSize, 1);
                    var y = new Matrix(OutputSize, 1);

                    for (int j = 0; j < InputSize; ++j)
                    {
                        x[j, 0] = line[j] / 255.0;
                    }
                    for (int j = 0; j < OutputSize; ++j)
                    {
                        y[j, 0] = line[InputSize + j];
                    }

                    TestData.Add(new Pair<Matrix, Matrix>(x, y));
                }
            }
            */
            TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));
            TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 1), MakeMatrix(1)));
            TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(1, 0), MakeMatrix(1)));
            TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));

            Console.WriteLine($"Training Loop : {N}");
            sw.Start();
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < TrainingData.Count; ++j)
                {
                    var output = BP.Train(TrainingData[j].First, TrainingData[j].Second, UseParallel);
                    Console.CursorLeft = 0;
                    if ((j + 1) % 500 == 0)
                        Console.Error.WriteLine($"{sw.Elapsed.ToString(@"hh\:mm\:ss")}[s]  All: {(int)(i * 100.0 / N)}%  Child Dataset: {(int)(j * 100.0 / TrainingData.Count)}%");
                }
            }
            sw.Stop();
            Console.WriteLine();
            Console.WriteLine($"Training Time : {sw.ElapsedMilliseconds.ToString()}[ms]");
            Console.WriteLine();

            Console.ReadKey();
            // Run
            for (int i = 0; i < TestData.Count; ++i)
            {
                var ret = BP.Forward(TestData[i].First);
                Console.WriteLine($"[{ret.Select(x => x.ToString()).ConcatWith(", ")}]  [{TestData[i].Second.Select(x => x.ToString()).ConcatWith(", ")}]");
            }

            Console.WriteLine();
            BP.Print(10);

            Console.ReadKey();
        }

        static Matrix MakeMatrix(double x, double y)
        {
            return new Matrix(new double[,] { { x }, { y } });
        }

        static Matrix MakeMatrix(double x)
        {
            return new Matrix(new double[,] { { x } });
        }
    }

    #region Pair
    public class Pair<T1, T2>
    {
        public T1 First;
        public T2 Second;
        public Pair() { First = default(T1); Second = default(T2); }
        public Pair(T1 f, T2 s) { First = f; Second = s; }
        public override string ToString() { return "(" + First + ", " + Second + ")"; }
        public override int GetHashCode() { return First.GetHashCode() ^ Second.GetHashCode(); }
        public override bool Equals(object obj)
        {
            if (ReferenceEquals(this, obj)) return true;
            else if (obj == null) return false;
            var tmp = obj as Pair<T1, T2>;
            return (object)tmp != null && First.Equals(tmp.First) && Second.Equals(tmp.Second);
        }
    }
    #endregion

    public static class Extentions
    {
        public static int MaxIndex(this IEnumerable<double> source)
        {
            return source.Select((v, i) => new { val = v, idx = i }).Aggregate((max, now) => (max.val > now.val) ? max : now).idx;
        }

        public static string ConcatWith<T>(this IEnumerable<T> source, string separator,
                string format, IFormatProvider provider = null) where T : IFormattable
        {
            return source.Select(x => x.ToString(format, provider)).Aggregate((a, b) => a + separator + b);
        }

        public static string ConcatWith<T>(this IEnumerable<T> source, string separator)
        {
            return string.Join(separator, source);
        }
    }
}
