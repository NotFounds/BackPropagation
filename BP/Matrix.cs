using System;
using System.Collections.Generic;
using System.Collections;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class Matrix : IEnumerable<double>
    {
        private double[] _Matrix;
        public int Row
        {
            private set;
            get;
        }
        public int Col
        {
            private set;
            get;
        }

        public Matrix(double[,] mat)
        {
            if (mat == null) throw new ArgumentNullException();

            Row = mat.GetLength(0);
            Col = mat.GetLength(1);
            _Matrix = new double[Row * Col];
            for (int i = 0; i < Row; ++i)
            {
                int row = i * Col;
                for (int j = 0; j < Col; ++j)
                {
                    _Matrix[row + j] = mat[i, j];
                }
            }
        }

        /// <summary>
        /// Matrix [row * col]
        /// </summary>
        /// <param name="row">Row</param>
        /// <param name="col">Col</param>
        public Matrix(int row, int col)
        {
            if (row <= 0 || col <= 0) throw new ArgumentException();

            Row = row;
            Col = col;
            _Matrix = new double[Row * Col];
        }

        /// <summary>
        /// Matrix [row * col]
        /// </summary>
        /// <param name="row">Row</param>
        /// <param name="col">Col</param>
        /// <param name="args">Matrix elements</param>
        public Matrix(int row, int col, params double[] args)
        {
            if (row <= 0 || col <= 0) throw new ArgumentException();
            if (args.Length != row * col) throw new ArgumentException();

            Row = row;
            Col = col;
            _Matrix = args;
        }

        public double this[int i, int j]
        {
            set
            {
                if (i < 0 || i >= Row) throw new IndexOutOfRangeException();
                if (j < 0 || j >= Col) throw new IndexOutOfRangeException();
                _Matrix[i * Col + j] = value;
            }
            get
            {
                if (i < 0 || i >= Row) throw new IndexOutOfRangeException();
                if (j < 0 || j >= Col) throw new IndexOutOfRangeException();
                return _Matrix[i * Col + j];
            }
        }

        public Matrix GetRow(int r)
        {
            if (r < 0 || r >= Row) throw new IndexOutOfRangeException();
            var ret = new Matrix(1, Col);

            for (int i = 0; i < Col; ++i)
            {
                ret[0, i] = this[0, i];
            }
            return ret;
        }

        public Matrix GetCol(int c)
        {
            if (c < 0 || c >= Col) throw new IndexOutOfRangeException();
            var ret = new Matrix(Row, 1);

            for (int i = 0; i < Row; ++i)
            {
                ret[i, 0] = this[i, 0];
            }
            return ret;
        }

        // O(N^2)
        public Matrix Transposition()
        {
            var ret = new Matrix(Col, Row);
            for (int i = 0; i < Row; ++i)
            {
                for (int j = 0; j < Col; ++j)
                {
                    ret[j, i] = this[i, j];
                }
            }
            return ret;
        }
        #region Arithmetic

        // O(N^2)
        public static Matrix operator +(Matrix a, Matrix b)
        {
            if (a.Row != b.Row || a.Col != b.Col) throw new ArithmeticException();
            var row = a.Row;
            var col = a.Col;
            var ret = new Matrix(row, col);

            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < col; ++j)
                {
                    ret[i, j] = a[i, j] + b[i, j];
                }
            }

            return ret;
        }

        // O(N^2)
        public static Matrix operator -(Matrix a, Matrix b)
        {
            if (a.Row != b.Row || a.Col != b.Col) throw new ArithmeticException();
            var row = a.Row;
            var col = a.Col;
            var ret = new Matrix(row, col);

            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < col; ++j)
                {
                    ret[i, j] = a[i, j] - b[i, j];
                }
            }
            return ret;
        }

        // O(N^2)
        public static Matrix operator *(double x, Matrix mat)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = x * mat[i, j];
                }
            }

            return ret;
        }

        // O(N^2)
        public static Matrix operator *(Matrix mat, double x)
        {
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = x * mat[i, j];
                }
            }

            return ret;
        }

        // O(N^2)
        public static Matrix operator /(Matrix mat, double x)
        {
            if (Math.Abs(x) < double.Epsilon) throw new DivideByZeroException();
            var ret = new Matrix(mat.Row, mat.Col);
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    ret[i, j] = mat[i, j] / x;
                }
            }

            return ret;
        }

        // O(N^3)
        public static Matrix operator *(Matrix a, Matrix b)
        {
            if (a.Col != b.Row) throw new ArithmeticException();
            var row = a.Row;
            var col = b.Col;
            var m = a.Col;
            var ret = new Matrix(row, col);

            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < col; ++j)
                {
                    for (int k = 0; k < m; ++k)
                    {
                        ret[i, j] += a[i, k] * b[k, j];
                    }
                }
            }

            return ret;
        }

        public static Matrix ParallelDot(Matrix a, Matrix b)
        {
            if (a.Col != b.Row) throw new ArithmeticException();
            var row = a.Row;
            var col = b.Col;
            var m = a.Col;
            var ret = new Matrix(row, col);

            Parallel.For(0, row, i =>
            {
                for (int j = 0; j < col; ++j)
                {
                    for (int k = 0; k < m; ++k)
                    {
                        ret[i, j] += a[i, k] * b[k, j];
                    }
                }
            });

            return ret;
        }
        #endregion

        #region Equal
        // O(N^2)
        public static bool operator ==(Matrix a, Matrix b)
        {
            if (a.Row != b.Row || a.Col != b.Col) return false;

            for (int i = 0; i < a.Row; ++i)
            {
                for (int j = 0; j < a.Col; ++j)
                {
                    if (!a[i, j].Equals(b[i, j])) return false;
                }
            }

            return true;
        }

        // O(N^2)
        public static bool operator !=(Matrix a, Matrix b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            var mat = obj as Matrix;
            return this == mat;
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
        #endregion

        #region Statistic
        public static Matrix Random(int row, int col, int? seed = null)
        {
            if (row <= 0 || col <= 0) throw new ArgumentException();
            var ret = new Matrix(row, col);
            var rnd = new Random(seed ?? DateTime.Now.Millisecond);
            for (int i = 0; i < row; ++i)
            {
                for (int j = 0; j < col; ++j)
                {
                    ret[i, j] = rnd.NextDouble() - rnd.NextDouble();
                }
            }
            return ret;
        }

        IEnumerator<double> IEnumerable<double>.GetEnumerator()
        {
            for (int i = 0; i < Row; ++i)
            {
                for (int j = 0; j < Col; ++j)
                {
                    yield return this[i, j];
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            for (int i = 0; i < Row; ++i)
            {
                for (int j = 0; j < Col; ++j)
                {
                    yield return this[i, j];
                }
            }
        }
        #endregion
    }
}
