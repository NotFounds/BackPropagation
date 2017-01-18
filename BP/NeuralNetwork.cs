using System;

namespace NeuralNetwork
{
    public enum LogisticFunctions
    {
        Sigmoid,
        SoftPlus,
        HypTan,
        ArcTan,
        IdentityFunction,
        ReLF,
        SoftMax
    }

    public enum LossFunctions
    {
        MSE,
        MLE,
        CE
    }

    public abstract class LogisticFunction
    {
        abstract public Matrix F(Matrix mat);
        abstract public void F(ref Matrix mat);
        abstract public double Df(double f, double x);
        abstract public Matrix Df(Matrix mat, int index);
    }

    public abstract class LossFunction
    {
        abstract public double F(double y, double t);
        abstract public double Df(double y, double t);
    }

    public static class NeuralNetwork
    {
        public static bool IsCanonicalLink(LogisticFunctions logistic, LossFunctions loss)
        {
            switch (logistic)
            {
                case LogisticFunctions.Sigmoid:
                    return loss == LossFunctions.MLE;
                case LogisticFunctions.IdentityFunction:
                    return loss == LossFunctions.MSE;
                case LogisticFunctions.SoftMax:
                    return loss == LossFunctions.CE;
                default:
                    return false;
            }
        }

        public static LogisticFunction GetLogisticFunction(LogisticFunctions logistic)
        {
            switch (logistic)
            {
                case LogisticFunctions.Sigmoid:
                    return new Sigmoid();
                case LogisticFunctions.SoftPlus:
                    return new SoftPlus();
                case LogisticFunctions.HypTan:
                    return new HyperbolicTan();
                case LogisticFunctions.ArcTan:
                    return new ArcTan();
                case LogisticFunctions.IdentityFunction:
                    return new IdentityFunction();
                case LogisticFunctions.ReLF:
                    return new RectifiedLinearUnit();
                case LogisticFunctions.SoftMax:
                    return new SoftMax();
                default:
                    return null;
            }
        }

        public static LossFunction GetLossFunction(LossFunctions loss)
        {
            switch (loss)
            {
                case LossFunctions.MSE:
                    return new MSE();
                case LossFunctions.MLE:
                    return new MLE();
                case LossFunctions.CE:
                    return new CE();
                default:
                    return null;
            }
        }
    }

    public class Sigmoid : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    ret[i, j] = 1.0 / (1.0 + Math.Exp(-mat[i, j]));
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    mat[i, j] = 1.0 / (1.0 + Math.Exp(-mat[i, j]));
        }

        public override double Df(double f, double x)
        {
            return f * (1.0 - f);
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override Matrix Df(Matrix mat, int index)
        {
            throw new NotImplementedException();
        }
    }

    public class SoftPlus : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    ret[i, j] = Math.Log(1 + Math.Exp(mat[i, j]));
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    mat[i, j] = Math.Log(1 + Math.Exp(mat[i, j]));
        }

        public override double Df(double f, double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override Matrix Df(Matrix mat, int index)
        {
            throw new NotImplementedException();
        }
    }

    public class HyperbolicTan : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    ret[i, j] = Math.Tanh(mat[i, j]);
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    mat[i, j] = Math.Tanh(mat[i, j]);
        }

        public override double Df(double f, double x)
        {
            return 1.0 - f * f;
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override Matrix Df(Matrix mat, int index)
        {
            throw new NotImplementedException();
        }
    }

    public class ArcTan : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    ret[i, j] = Math.Atan(mat[i, j]);
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    mat[i, j] = Math.Atan(mat[i, j]);
        }

        public override double Df(double f, double x)
        {
            return 1.0 / (1.0 + x * x);
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override Matrix Df(Matrix mat, int index)
        {
            throw new NotImplementedException();
        }
    }

    public class IdentityFunction : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            return mat;
        }

        public override void F(ref Matrix mat)
        {
        }

        public override double Df(double f, double x)
        {
            return 1.0;
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override Matrix Df(Matrix mat, int index)
        {
            throw new NotImplementedException();
        }
    }

    public class RectifiedLinearUnit : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    ret[i, j] = Math.Max(0, mat[i, j]);
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
                for (int j = 0; j < mat.Col; ++j)
                    mat[i, j] = Math.Max(0, mat[i, j]);
        }

        public override double Df(double f, double x)
        {
            return (x >= 0) ? 1 : 0;
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override Matrix Df(Matrix mat, int index)
        {
            throw new NotImplementedException();
        }
    }

    public class SoftMax : LogisticFunction
    {
        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int j = 0; j < mat.Col; ++j)
            {
                var sum = 0.0;
                var max = 0.0;
                for (int i = 0; i < mat.Row; ++i)
                    max = Math.Max(mat[i, j], max);
                for (int i = 0; i < mat.Row; ++i)
                    sum += Math.Exp(mat[i, j] - max);
                for (int i = 0; i < mat.Row; ++i)
                    ret[i, j] = Math.Exp(mat[i, j] - max) / sum;
            }
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int j = 0; j < mat.Col; ++j)
            {
                var sum = 0.0;
                var max = 0.0;
                for (int i = 0; i < mat.Row; ++i)
                    max = Math.Max(mat[i, j], max);
                for (int i = 0; i < mat.Row; ++i)
                    sum += Math.Exp(mat[i, j] - max);
                for (int i = 0; i < mat.Row; ++i)
                    mat[i, j] = Math.Exp(mat[i, j] - max) / sum;
            }
        }

        /// <summary>
        /// No Implemented : Don's Use Me
        /// </summary>
        public override double Df(double f, double x)
        {
            throw new NotImplementedException();
        }

        public override Matrix Df(Matrix mat, int index)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int j = 0; j < mat.Col; ++j)
                for (int i = 0; i < mat.Row; ++i)
                    ret[i, j] = (i == index) ? mat[i, j] * (1 - mat[i, j]) : mat[index, j] * -mat[i, j];
            return ret;
        }
    }

    // Mean-Squared-Error lose function : for Regression
    public class MSE : LossFunction
    {
        public override double F(double y, double t)
        {
            return (y - t) * (y - t) / 2;
        }

        public override double Df(double y, double t)
        {
            return y - t;
        }
    }

    // Maximum-Likelihood-Estimation lose function : for Binary Classifications
    public class MLE : LossFunction
    {
        public override double F(double y, double t)
        {
            return -t * Math.Log(y) - (1 - t) * Math.Log(1 - y);
        }

        public override double Df(double y, double t)
        {
            return (y - t) / (y * (1 - y));
        }
    }

    // Cross-Entropy lose function : for Multi-class Classifications
    public class CE : LossFunction
    {
        public override double F(double y, double t)
        {
            return -t * Math.Log(y);
        }

        public override double Df(double y, double t)
        {
            return -t / y;
        }
    }
}
