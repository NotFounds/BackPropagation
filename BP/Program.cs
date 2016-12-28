using System;
using System.Diagnostics;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

using NeuralNetwork;
using NeuralNetwork.BackPropagation;

namespace NotFounds
{
    class Program
    {
        static void Main(string[] args)
        {
            var sw = new Stopwatch();

            const int    N    = 2000;
            const double Rate = 0.5;

            // 2 Input
            // 3 Hidden
            // 2 Output

            //  Input Weight: 3 * 2 Matrix
            // Output Weight: 2 * 3 Matrix

            // Random weight/layer
            var inputWeight  = new Matrix(new double[,] { { 0.1, 0.3 }, { -0.2, 0.4 }, { 0.2, -0.3} });
            var outputWeight = new Matrix(new double[,] { { 0.5, 0.2, 0.3 }, { -0.4, 0.1, 0.2 } });
            var hiddenLayer  = new Matrix(new double[,] { { 0.2 }, { -0.3 }, { 0.1} });
            var outputLayer  = new Matrix(new double[,] { { 0.4 }, { 0.2} });

            var BP = new BackPropagation(inputWeight, outputWeight, hiddenLayer, outputLayer, Rate);

            Console.WriteLine($"Training Loop : {N}");
            sw.Start();
            for (int i = 0; i < N; ++i)
            {
                // XOR AND
                BP.Train(MakeMatrix(0, 0), MakeMatrix(0, 0));
                BP.Train(MakeMatrix(0, 1), MakeMatrix(1, 0));
                BP.Train(MakeMatrix(1, 0), MakeMatrix(1, 0));
                BP.Train(MakeMatrix(1, 1), MakeMatrix(0, 1));
            }
            sw.Stop();
            Console.WriteLine($"Training Time : {sw.ElapsedMilliseconds.ToString()}[ms]");

            // Run
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    var ret = BP.Run(MakeMatrix(i, j));
                    Console.WriteLine($"{i} xor {j} = {ret[0, 0].ToString("F20").Substring(0, 15)}  {i} and {j} = {ret[1, 0].ToString("F20").Substring(0, 15)}");
                }
            }
            Console.WriteLine();
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    var ret = BP.Run(MakeMatrix(i, j));
                    Console.WriteLine($"{i} xor {j} = {(int)(ret[0, 0] + 0.5)}  {i} and {j} = {(int)(ret[1, 0] + 0.5)}");
                }
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
}
