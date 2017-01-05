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

            const int N                   = 10000;
            const double Rate             = 0.3;

            const string TrainingDataFile = null;
            const string TestDataFile     = null;
            const int TrainingDataNum     = 0;
            const int TestDataNum         = 0;

            const int InputSize           = 2;
            const int HiddenSize          = 3;
            const int OutputSize          = 1;

            // Random weight/layer
            var inputWeight = Matrix.Random(HiddenSize, InputSize);   //new Matrix(new double[,] { { 0.1, 0.3 }, { -0.2, 0.4 }, { 0.2, -0.3} });
            var outputWeight = Matrix.Random(OutputSize, HiddenSize); //new Matrix(new double[,] { { 0.5, 0.2, 0.3 }, { -0.4, 0.1, 0.2 } });
            var hiddenLayer = Matrix.Random(HiddenSize, 1);           //new Matrix(new double[,] { { 0.2 }, { -0.3 }, { 0.1} });
            var outputLayer = Matrix.Random(OutputSize, 1);           //new Matrix(new double[,] { { 0.4 }, { 0.2} });

            var BP = new BackPropagation(inputWeight, outputWeight, hiddenLayer, outputLayer, null, Rate);

            Console.WriteLine($"Initial State");
            BP.Print(4);
            Console.WriteLine();

            // Input Training Data from File
            var TrainingData = new List<Pair<Matrix, Matrix>>();
            /*
               using (var sr = new StreamReader(Path.GetDirectoryName(TrainingDataFile)))
               {
               for (int i = 0; i < TrainingDataNum; ++i)
               {
               var line = sr.ReadLine().Split(',').Select(s => double.Parse(s)).ToArray();
               var x = new Matrix(InputSize, 1);
               var y = new Matrix(OutputSize, 1);

               for (int j = 0; j < InputSize; ++j)
               {
               x[j, 0] = line[j + 1];
               }
               y[0, 0] = line[0];

               TrainingData.Add(new Pair<Matrix, Matrix>(x, y));
               }
               }
               */

            // Training Data
            {
                TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));
                TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 1), MakeMatrix(1)));
                TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(1, 0), MakeMatrix(1)));
                TrainingData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));
            }

            // Input Test Data from File
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
               x[j, 0] = line[j + 1];
               }
               y[0, 0] = line[0];

               TestData.Add(new Pair<Matrix, Matrix>(x, y));
               }
               }
               */

            // Test Data
            {
                TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));
                TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 1), MakeMatrix(1)));
                TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(1, 0), MakeMatrix(1)));
                TestData.Add(new Pair<Matrix, Matrix>(MakeMatrix(0, 0), MakeMatrix(0)));
            }

            Console.WriteLine($"Training Loop : {N}");
            sw.Start();
            for (int i = 0; i < N; ++i)
            {
                Console.CursorLeft = 0;
                Console.Write($"{sw.Elapsed.ToString(@"hh\:mm\:ss")}[s]  {(int)((i + 1) * 100.0 / N)}%");

                for (int j = 0; j < TrainingData.Count; ++j)
                {
                    BP.Train(TrainingData[j].First, TrainingData[j].Second);
                }
            }
            sw.Stop();
            Console.WriteLine();
            Console.WriteLine($"Training Time : {sw.ElapsedMilliseconds.ToString()}[ms]");
            Console.WriteLine();

            // Run
            Console.WriteLine("  Result  | Truth");
            for (int i = 0; i < TestData.Count; ++i)
            {
                var ret = BP.Run(TestData[i].First);
                Console.WriteLine($"{ret[0, 0].ToString("F22").Substring(0, 9)} |   {TestData[i].Second[0, 0]}");
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
        public T1 First { get; set; }
        public T2 Second { get; set; }
        public Pair() { First = default(T1); Second = default(T2); }
        public Pair(T1 f, T2 s) { First = f; Second = s; }
        public override string ToString() { return "(" + First + ", " + Second + ")"; }
        public override int GetHashCode() { return First.GetHashCode() ^ Second.GetHashCode(); }
        public override bool Equals(object obj)
        {
            if (obj == null) return false;
            if (ReferenceEquals(this, obj)) return true;
            var tmp = obj as Pair<T1, T2>;
            return (object)tmp != null && First.Equals(tmp.First) && Second.Equals(tmp.Second);
        }
    }
#endregion
}
