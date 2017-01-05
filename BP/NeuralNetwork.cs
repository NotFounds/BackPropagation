using System;

namespace NeuralNetwork
{
    public abstract class LogisticFunction
    {
        abstract public double F(double x);
        abstract public Matrix F(Matrix mat);
        abstract public void F(ref Matrix mat);
        abstract public double Df(double f, double x);
    }

    public class Sigmoid : LogisticFunction
    {
        public override double F(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public override Matrix F(Matrix mat)
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

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = 1.0 / (1.0 + Math.Exp(-mat[i, j]));
                }
            }
        }

        public override double Df(double f, double x)
        {
            return f * (1.0 - f);
        }
    }

    public class SoftPlus : LogisticFunction
    {
        public override double F(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public override Matrix F(Matrix mat)
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

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = Math.Log(1 + Math.Exp(mat[i, j]));
                }
            }
        }

        public override double Df(double f, double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
    }

    public class HyperbolicTan : LogisticFunction
    {
        public override double F(double x)
        {
            return Math.Tanh(x);
        }

        public override Matrix F(Matrix mat)
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

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = Math.Tanh(mat[i, j]);
                }
            }
        }

        public override double Df(double f, double x)
        {
            return 1.0 - f * f;
        }
    }

    public class ArcTan : LogisticFunction
    {
        public override double F(double x)
        {
            return Math.Atan(x);
        }

        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = Math.Atan(mat[i, j]);
                }
            }
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = Math.Atan(mat[i, j]);
                }
            }
        }

        public override double Df(double f, double x)
        {
            return 1.0 / (1.0 + x * x);
        } 
    }

    public class IdentityFunction : LogisticFunction
    {
        public override double F(double x)
        {
            return x;
        }

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
    }

    public class RectifiedLinearUnit : LogisticFunction
    {
        public override double F(double x)
        {
            return Math.Max(0, x);
        }

        public override Matrix F(Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = Math.Max(0, mat[i, j]);
                }
            }
            return ret;
        }

        public override void F(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = Math.Max(0, mat[i, j]);
                }
            }
        }

        public override double Df(double f, double x)
        {
            return (x >= 0) ? 1 : 0;
        }
    }
}
