using System;

namespace NeuralNetwork
{
    public abstract class LogisticFunction
    {
        abstract public double Caluculate(double x);
        abstract public Matrix Caluculate(Matrix mat);
        abstract public void Caluculate(ref Matrix mat);
    }

    public class Sigmoid : LogisticFunction
    {
        public override double Caluculate(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public override Matrix Caluculate(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = 1.0 / (1.0 + Math.Exp(-mat[i, j]));
                }
            }
            return ret;
        }

        public override void Caluculate(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = 1.0 / (1.0 + Math.Exp(-mat[i, j]));
                }
            }
        }
    }

    public class SoftSign : LogisticFunction
    {
        public override double Caluculate(double x)
        {
            return x / (1 + Math.Abs(x));
        }

        public override Matrix Caluculate(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = mat[i, j] / (1 + Math.Abs(mat[i, j]));
                }
            }
            return ret;
        }

        public override void Caluculate(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = mat[i, j] / (1 + Math.Abs(mat[i, j]));
                }
            }
        }
    }


    public class SoftPlus : LogisticFunction
    {
        public override double Caluculate(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public override Matrix Caluculate(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = Math.Log(1 + Math.Exp(mat[i, j]));
                }
            }
            return ret;
        }

        public override void Caluculate(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = Math.Log(1 + Math.Exp(mat[i, j]));
                }
            }
        }
    }

    public class HyperbolicTan : LogisticFunction
    {
        public override double Caluculate(double x)
        {
            return Math.Tanh(x);
        }

        public override Matrix Caluculate(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = Math.Tanh(mat[i, j]);
                }
            }
            return ret;
        }

        public override void Caluculate(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = Math.Tanh(mat[i, j]);
                }
            }
        }
    }

    public class IdentityFunction : LogisticFunction
    {
        public override double Caluculate(double x)
        {
            return x;
        }

        public override Matrix Caluculate(Matrix mat)
        {
            return mat;
        }

        public override void Caluculate(ref Matrix mat)
        {
        }
    }

    public class StepFunction : LogisticFunction
    {
        public override double Caluculate(double x)
        {
            return (x > 0) ? 1 : 0;
        }

        public override Matrix Caluculate(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = (mat[i, j] > 0) ? 1 : 0;
                }
            }
            return ret;
        }

        public override void Caluculate(ref Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = (mat[i, j] > 0) ? 1 : 0;
                }
            }
        }
    }
}
