using System;

namespace NeuralNetwork
{
    public static class LogisticFunctions
    {
        public static double Step(double x, double threshold = 0.0)
        {
            return (x < threshold) ? 0 : 1;
        }

        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double SoftSign(double x)
        {
            return x / (1 + Math.Abs(x));
        }

        public static double SoftPlus(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public static double HyperbolicTan(double x)
        {
            return Math.Tanh(x);
        }
    }
}
