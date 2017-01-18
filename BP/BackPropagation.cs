using System;

using NeuralNetwork;
using static NeuralNetwork.NeuralNetwork;

namespace NeuralNetwork.BackPropagation
{
    public class BackPropagation
    {
        public Matrix _hiddenLayer { private set; get; }
        public Matrix _outputLayer { private set; get; }
        public Matrix _inputWeight { private set; get; }
        public Matrix _outputWeight { private set; get; }
        public double LearnRate { set; get; }
        public LogisticFunctions  HiddenLogisticFunc { private set; get; }
        public LogisticFunctions  OutputLogisticFunc { private set; get; }
        public LossFunctions      LossFunc { private set; get; }
        private LogisticFunction _hiddenLogisticFunc;
        private LogisticFunction _outputLogisticFunc;
        private LossFunction     _lossFunc;

        /// <summary>
        /// 3Layer Backpropagation Class
        /// </summary>
        /// <param name="inputWeight">Input Weight.</param>
        /// <param name="outputWeight">Output Weight.</param>
        /// <param name="hiddenLayer">Hidden layer.</param>
        /// <param name="outputLayer">Output layer.</param>
        /// <param name="hiddenLogisticFunc">Hidden Layer Logistic Function. Default = SigmoidFunc</param>
        /// <param name="outputLogisticFunc">Output Layer Logistic Function. Default = SigmoidFunc</param>
        /// <param name="lossFunc">Output Layer Loss Function. Default = MSE</param>
        /// <param name="learnRate">Learn rate. Default = 0.01</param>
        public BackPropagation(Matrix inputWeight, Matrix outputWeight, Matrix hiddenLayer, Matrix outputLayer,
                               LogisticFunctions hiddenLogisticFunc = LogisticFunctions.Sigmoid,
                               LogisticFunctions outputLogisticFunc = LogisticFunctions.Sigmoid,
                               LossFunctions lossFunc = LossFunctions.MSE, double learnRate = 0.01)
        {
            _inputWeight = inputWeight;
            _outputWeight = outputWeight;
            _hiddenLayer = hiddenLayer;
            _outputLayer = outputLayer;

            HiddenLogisticFunc = hiddenLogisticFunc;
            OutputLogisticFunc = outputLogisticFunc;
            LossFunc = lossFunc;

            _hiddenLogisticFunc = GetLogisticFunction(hiddenLogisticFunc);
            _outputLogisticFunc = GetLogisticFunction(outputLogisticFunc);
            _lossFunc = GetLossFunction(lossFunc);

            LearnRate = learnRate;
        }

        public Matrix Forward(Matrix input)
        {
            var hidden = _inputWeight * input + _hiddenLayer;
            _hiddenLogisticFunc.F(ref hidden);

            var output = _outputWeight * hidden + _outputLayer;
            _outputLogisticFunc.F(ref output);

            return output;
        }

        public Matrix Train(Matrix input, Matrix target, bool useParallel = false)
        {
            var hidden = (useParallel ? Matrix.ParallelDot(_inputWeight, input) : _inputWeight * input) + _hiddenLayer;
            var hiddenF = _hiddenLogisticFunc.F(hidden);

            var output = (useParallel ? Matrix.ParallelDot(_outputWeight, hiddenF) : _outputWeight * hiddenF) + _outputLayer;
            var outputF = _outputLogisticFunc.F(output);

            // Calculate the error
            var outputAdjustment = new Matrix(outputF.Row, 1);
            var hiddenAdjustment = new Matrix(hiddenF.Row, 1);

            if (IsCanonicalLink(OutputLogisticFunc, LossFunc))
            {
                outputAdjustment = outputF - target;
            }
            else
            {
                for (int i = 0; i < outputAdjustment.Row; ++i)
                {
                    outputAdjustment[i, 0] = _outputLogisticFunc.Df(outputF[i, 0], output[i, 0]) * _lossFunc.Df(outputF[i, 0], target[i, 0]);
                }
            }

            for (int i = 0; i < hiddenAdjustment.Row; ++i)
            {
                for (int j = 0; j < outputAdjustment.Row; ++j)
                {
                    hiddenAdjustment[i, 0] += _hiddenLogisticFunc.Df(hiddenF[j, 0], hidden[j, 0]) * outputAdjustment[j, 0] * _outputWeight[j, i];
                }
            }

            // Adjust the weights
            for (int i = 0; i < _inputWeight.Row; ++i)
            {
                for (int j = 0; j < _inputWeight.Col; ++j)
                {
                    _inputWeight[i, j] -= hiddenAdjustment[i, 0] * input[j, 0] * LearnRate;
                }
            }

            for (int i = 0; i < _outputWeight.Row; ++i)
            {
                for (int j = 0; j < _outputWeight.Col; ++j)
                {
                    _outputWeight[i, j] -= outputAdjustment[i, 0] * hiddenF[j, 0] * LearnRate;
                }
            }

            _hiddenLayer -= hiddenAdjustment * LearnRate;
            _outputLayer -= outputAdjustment * LearnRate;

            return outputF;
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
