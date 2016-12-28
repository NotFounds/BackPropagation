using System;

using NeuralNetwork;

namespace NeuralNetwork.BackPropagation
{
    public class BackPropagation
    {
        public Matrix _hiddenLayer { private set; get; }
        public Matrix _outputLayer { private set; get; }
        public Matrix _inputWeight { private set; get; }
        public Matrix _outputWeight { private set; get; }
        public double LearnRate { set; get; }

        /// <summary>
        /// 3Layer Backpropagation Class
        /// </summary>
        /// <param name="inputWeight">Input Weight.</param>
        /// <param name="outputWeight">Output Weight.</param>
        /// <param name="hiddenLayer">Hidden layer.</param>
        /// <param name="outputLayer">Output layer.</param>
        /// <param name="learnRate">Learn rate. Default = 0.001</param>
        public BackPropagation(Matrix inputWeight, Matrix outputWeight, Matrix hiddenLayer, Matrix outputLayer, double learnRate = 0.001)
        {
            _inputWeight = inputWeight;
            _outputWeight = outputWeight;
            _hiddenLayer = hiddenLayer;
            _outputLayer = outputLayer;
            LearnRate = learnRate;
        }

        public Matrix Run(Matrix input)
        {
            var outHidden = _inputWeight * input + _hiddenLayer;
            Sigmoid(ref outHidden);

            var output = _outputWeight * outHidden + _outputLayer;
            Sigmoid(ref output);

            return output;
        }

        public void Train(Matrix input, Matrix target)
        {
            var outHidden = _inputWeight * input + _hiddenLayer;
            Sigmoid(ref outHidden);

            var output = _outputWeight * outHidden + _outputLayer;
            Sigmoid(ref output);

            // Calculate the error
            var outputAdjustment = new Matrix(output.Row, 1);
            var hiddenAdjustment = new Matrix(outHidden.Row, 1);

            for (int i = 0; i < outputAdjustment.Row; ++i)
            {
                outputAdjustment[i, 0] = output[i, 0] * (1 - output[i, 0]) * (target[i, 0] - output[i, 0]);
            }

            for (int i = 0; i < hiddenAdjustment.Row; ++i)
            {
                for (int j = 0; j < outputAdjustment.Row; ++j)
                {
                    hiddenAdjustment[i, 0] += outHidden[j, 0] * (1 - outHidden[j, 0]) * outputAdjustment[j, 0] * _outputWeight[j, i];
                }
            }

            // Adjust the weights
            for (int i = 0; i < _inputWeight.Row; ++i)
            {
                for (int j = 0; j < _inputWeight.Col; ++j)
                {
                    _inputWeight[i, j] += hiddenAdjustment[i, 0] * input[j, 0] * LearnRate;
                }
            }

            for (int i = 0; i < _outputWeight.Row; ++i)
            {
                for (int j = 0; j < _outputWeight.Col; ++j)
                {
                    _outputWeight[i, j] += outputAdjustment[i, 0] * outHidden[j, 0] * LearnRate;
                }
            }

            for (int i = 0; i < _hiddenLayer.Row; ++i)
            {
                _hiddenLayer[i, 0] += hiddenAdjustment[i, 0] * LearnRate;
            }

            _outputLayer += outputAdjustment * LearnRate;
        }

        public void Sigmoid(ref Matrix mat)
        {
            for (int i = 0; i < mat.Row; ++i)
            {
                for (int j = 0; j < mat.Col; ++j)
                {
                    mat[i, j] = LogisticFunctions.Sigmoid(mat[i, j]);
                }
            }
        }

        public void Print(int n = 20)
        {
            Console.WriteLine($"Learnrate: {LearnRate}");
            PrintWeight(n);
            PrintLayer(n);
        }

        public void PrintWeight(int n = 20)
        {
            Console.WriteLine("Input Weight");
            for (int i = 0; i < _inputWeight.Row; ++i)
            {
                for (int j = 0; j < _inputWeight.Col; ++j)
                {
                    Console.Write($"{_inputWeight[i, j].ToString("F22").Substring(0, n)}  ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();

            Console.WriteLine("Output Weight");
            for (int i = 0; i < _outputWeight.Row; ++i)
            {
                for (int j = 0; j < _outputWeight.Col; ++j)
                {
                    Console.Write($"{_outputWeight[i, j].ToString("F22").Substring(0, n)}  ");
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        public void PrintLayer(int n = 20)
        {
            Console.WriteLine("Hidden Layer");
            for (int i = 0; i < _hiddenLayer.Row; ++i)
            {
                Console.WriteLine(_hiddenLayer[i, 0].ToString("F22").Substring(0, n));
            }
            Console.WriteLine();

            Console.WriteLine("Output Layer");
            for (int i = 0; i < _outputLayer.Row; ++i)
            {
                Console.WriteLine(_outputLayer[i, 0].ToString("F22").Substring(0, n));
            }
            Console.WriteLine();
        }
    }
}
