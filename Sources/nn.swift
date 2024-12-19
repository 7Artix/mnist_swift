import Foundation

//单节点结构体
struct NodeStructure {
    var activationFunction: ActivationFunction
    var weightInitializer: (Int?, Int?) -> Double
    var bias: Double
}

//输出层结构体
struct OutputLayer {
    var outputSize: Int
    //输出层(激活值): 神经网络输出的未经激活的线性叠加值
    var valueNetwork: [Double]
    //输出层(归一值): 归一化后值
    var valueNormalized: [Double]
    //对损失函数求导梯度
    var dLoss: [Double]
    //对归一化函数求导梯度
    var dNormalization: [Double]
    //归一化函数
    var normalizationFunction: NormalizationFunction
    //损失函数
    var lossFunction: LossFunction

    init(outputSize: Int, normalizationFunction: NormalizationFunction, lossFunction: LossFunction) {
        self.outputSize = outputSize
        self.valueNetwork = Array(repeating: Double(0.0), count: self.outputSize)
        self.valueNormalized = Array(repeating: Double(0.0), count: self.outputSize)
        self.dLoss = Array(repeating: Double(0.0), count: self.outputSize)
        self.dNormalization = Array(repeating: Double(0.0), count: self.outputSize)
        self.normalizationFunction = normalizationFunction
        self.lossFunction = lossFunction
    }
}

//神经网络配置结构体
struct NNConfig {
    //输入层参数数量, 本层并无实际节点, 也不包括激活等操作
    let inputSize: Int
    //神经网络结构, 配置层数, 激活函数, 初始化值等
    var structure: [[NodeStructure]]
    //输出层结构
    var outputLayer: OutputLayer

    init(inputSize: Int, structure: [[NodeStructure]], outputLayer: OutputLayer) {
        guard let outputSizeFromStructure = structure.last?.count else {
            fatalError("Error: NNConfig. Incorrect input node structure")
        }
        if (outputLayer.outputSize != outputSizeFromStructure) {
            fatalError("Error: NNConfig. Output layer doesn't match the network")
        }
        self.inputSize = inputSize
        self.structure = structure
        self.outputLayer = outputLayer
    }
}

//训练配置
struct TrainingConfig {
    //单个batch的样本数量
    var batchSize: Int
    //学习率
    var learningRate: Double
    //损失函数始终持续低于历史最佳时, 最多尝试次数
    var negativeAttempts: Int

    init(batchSize: Int, learningRate: Double, negativeAttempts: Int) {
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.negativeAttempts = negativeAttempts
    }
}

class NN {
    //网络参数结构体
    struct NNParameter {
        var weights: [[[Double]]]
        var biases: [[Double]]
    }
    //训练配置
    var trainingConfig: TrainingConfig
    //权重张量: 3维
    var weights: [[[Double]]]
    //权重矩阵
    var biases: [[Double]]
    //线性叠加值矩阵
    var zs: [[Double]]
    //激活值矩阵
    var activations: [[Double]]
    //激活函数矩阵
    var activationFunctions: [[ActivationFunction]]
    //输出层结构体
    var outputLayer: OutputLayer
    //输入层节点数量(输入层不包含在网络结构中)
    var inputSize: Int
    //输出层节点数量
    var outputSize: Int
    //层结构, 存储每层节点的数量, 便于初始化
    var layerStructure: [Int]
    //权重结构, 存储每个节点的权重数量, 便于初始化
    var weightStructure: [[Int]]
    //误差梯度, 对各层线性叠加值z
    var dZ: [[Double]]
    //相对权重张量的梯度
    var dW: [[[Double]]]
    //相对偏置矩阵的梯度
    var dB: [[Double]]
    //batch多样本梯度存储
    var dWeightsBatch: [[[[Double]]]] = []
    var dBiasesBatch: [[[Double]]] = []
    //Batch平均梯度
    var dParametersMean: NN.NNParameter?
    //历史最佳参数
    var historyBest: NNParameter?
    //上次前向传播的预测结果
    var predictionLast: Int?
    //上次前向传播的损失值
    var lossLast: Double?
    //上次输入值
    var inputLast: [Double]?
    //上次标签
    var labelsLast: [Double]?
    //标签含义
    var labelsMeaning: [String]?

    init(networkConfig: NNConfig, trainingConfig: TrainingConfig){
        self.trainingConfig = trainingConfig
        self.outputLayer = networkConfig.outputLayer
        self.inputSize = networkConfig.inputSize
        self.outputSize = self.outputLayer.outputSize
        //从网络结构获取层结构
        self.layerStructure = networkConfig.structure.map {$0.count}
        //层结构加入输入层数量, 但并无实际节点, 仅便于后续建立连接赋权重
        self.layerStructure.insert(networkConfig.inputSize, at: 0)
        //layerStructure结构: [input层节点数量 : 隐含层1节点数量 : ... : 隐含层n节点数量 : output层节点数量]

        //建立权重结构, 多加入了从输出层到归一化层的无效权重
        self.weightStructure = []
        for index in 0..<(self.layerStructure.count - 1) {
            let nodeCount = self.layerStructure[index]
            let nextLayerNodeCount = self.layerStructure[index + 1]
            self.weightStructure.append(Array(repeating: nodeCount, count: nextLayerNodeCount))
        }
        //初始化
        self.zs = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.activations = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.activationFunctions = self.layerStructure.dropFirst().map { Array(repeating: ReLU(), count: $0)}
        self.biases = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.weights = self.weightStructure.map { node in node.map {Array(repeating: 0.0, count: $0)}}
        self.dZ = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.dB = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.dW = self.weightStructure.map { node in node.map {Array(repeating: 0.0, count: $0)}}
        //根据配置初始化权重值, 偏置值与激活函数
        for (indexLayer, layer) in networkConfig.structure.enumerated() {
            let inputNodeCount = layerStructure[indexLayer]
            let outputNodeCount = indexLayer + 2 < layerStructure.count ? layerStructure[indexLayer + 2] : layerStructure[indexLayer + 1]
            for (indexNode, node) in layer.enumerated() {
                biases[indexLayer][indexNode] = node.bias
                activationFunctions[indexLayer][indexNode] = node.activationFunction
                for indexWeight in 0..<weights[indexLayer][indexNode].count {
                    weights[indexLayer][indexNode][indexWeight] = node.weightInitializer(inputNodeCount, outputNodeCount)
                }
            }
        }
    }

    func fp(input: [Double], labels: [Double]) {
        if input.count != self.inputSize {
            fatalError("Error: FP. Input data doesn't match the network")
        }
        if labels.count != self.outputSize {
            fatalError("Error: FP. Input labels doesn't match the network")
        }
        self.labelsLast = labels
        self.inputLast = input
        var activationsPreviousLayer = input
        for(indexLayer, weightsLayer) in self.weights.enumerated() {
            let countNode = self.activations[indexLayer].count
            for indexNode in 0..<countNode {
                let weightsNode = weightsLayer[indexNode]
                let weightedSum = zip(weightsNode, activationsPreviousLayer).reduce(0.0) { sum, pair in sum + pair.0 * pair.1} + self.biases[indexLayer][indexNode]
                self.zs[indexLayer][indexNode] = weightedSum
                self.activations[indexLayer][indexNode] = self.activationFunctions[indexLayer][indexNode].forward(weightedSum)
            }
            activationsPreviousLayer = self.activations[indexLayer]
        }
        guard let valueNetwork = zs.last else {
            fatalError("Error: FP. Invalid output layer values")
        }
        self.outputLayer.valueNetwork = valueNetwork
        for (index, _) in self.outputLayer.valueNetwork.enumerated() {
            self.outputLayer.valueNormalized[index] = self.outputLayer.normalizationFunction.forward(inputAll: self.outputLayer.valueNetwork, indexNode: index)
        }
        self.lossLast = self.outputLayer.lossFunction.forward(predictions: self.outputLayer.valueNormalized, labels: labels)
    }

    func bp() {
        guard let ValuesInput = self.inputLast else {
            fatalError("Error: BP. Call FP first")
        }
        //计算对输出层梯度
        for index in 0..<self.outputSize {
            self.outputLayer.dLoss[index] = self.outputLayer.lossFunction.backward(indexNode: index)
        }
        self.outputLayer.lossFunction.reset()
        for index in 0..<self.outputSize {
            self.outputLayer.dNormalization[index] = self.outputLayer.normalizationFunction.backward(dInputAll: self.outputLayer.dLoss, indexNode: index)
        }
        self.dZ[self.dZ.count - 1] = self.outputLayer.dNormalization
        //var gradientsFromPreviousLayer = self.outputLayer.dNormalization
        //计算除输出层外的dZ
        for indexLayer in (0..<self.dZ.count - 1).reversed() {
            for indexWeight in 0..<self.weights[indexLayer+1][0].count {
                var dZElement: Double = 0.0
                for indexNode in 0..<self.weights[indexLayer+1].count {
                    dZElement += self.weights[indexLayer+1][indexNode][indexWeight] * self.dZ[indexLayer+1][indexNode]
                }
                self.dZ[indexLayer][indexWeight] = dZElement * self.activationFunctions[indexLayer][indexWeight].backward(self.zs[indexLayer][indexWeight])
            }
        }
        //∂Loss/∂Biases = ∂Loss/∂Z
        self.dB = self.dZ
        //∂Loss/∂Wn = ∂Loss/∂Z * a(n-1)^T
        for indexLayer in 1..<self.weights.count {
            for indexNode in 0..<self.dZ[indexLayer].count {
                for indexWeight in 0..<self.activations[indexLayer-1].count {
                    self.dW[indexLayer][indexNode][indexWeight] = self.dZ[indexLayer][indexNode] * self.activations[indexLayer - 1][indexWeight]
                }
            }
        }
        for indexNode in 0..<self.weights[0].count {
            for indexWeight in 0..<self.weights[0][indexNode].count {
                self.dW[0][indexNode][indexWeight] = self.dZ[0][indexNode] * ValuesInput[indexWeight]
            }
        }
        inputLast = nil
        self.zs = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.activations = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
    }

    func resetGradients() {
        for indexLayer in 0..<self.dW.count {
            for indexNode in 0..<self.dW[indexLayer].count {
                self.dW[indexLayer][indexNode] = Array(repeating: Double(0.0), count: self.dW[indexLayer][indexNode].count)
            }
            self.dB[indexLayer] = Array(repeating: Double(0.0), count: self.dB[indexLayer].count)
        }
    }

    func updateParameters(withGradients parameters: NNParameter) {
        for indexLayer in 0..<parameters.weights.count {
            for indexNode in 0..<parameters.weights[indexLayer].count {
                for indexToPreviousNode in 0..<parameters.weights[indexLayer][indexNode].count {
                    self.weights[indexLayer][indexNode][indexToPreviousNode] -= self.trainingConfig.learningRate * parameters.weights[indexLayer][indexNode][indexToPreviousNode]
                }
                self.biases[indexLayer][indexNode] -= self.trainingConfig.learningRate * parameters.biases[indexLayer][indexNode]
            }
        }
    }

    func descentSingleStep() {
        let dParameters = NN.NNParameter(weights: self.dW, biases: self.dB)
        updateParameters(withGradients: dParameters)
    }

    func descentbatches(inputs: [[Double]], labels: [[Double]]) {
        if inputs.count != self.trainingConfig.batchSize || labels.count != self.trainingConfig.batchSize {
            fatalError("Error: Gradient descent. Input Data doesn't match batch size")
        }
        self.dWeightsBatch = []
        self.dBiasesBatch = []
        for indexBatch in 0..<self.trainingConfig.batchSize {
            self.fp(input: inputs[indexBatch], labels: labels[indexBatch])
            print("Loss: \(self.getLoss())")
            self.bp()
            self.dWeightsBatch.append(self.dW)
            self.dBiasesBatch.append(self.dB)
            self.resetGradients()
        }
        var dParametersBatchesMean = NNParameter(
            weights: self.dW.map { layer in
                layer.map { Array(repeating: Double(0.0), count: $0.count) }
            },
            biases: self.dB.map { layer in
                Array(repeating: Double(0.0), count: layer.count)
            }
        )
        for indexLayer in 0..<self.dW.count {
            for indexNode in 0..<self.dW[indexLayer].count {
                dParametersBatchesMean.weights[indexLayer][indexNode] = Array(repeating: Double(0.0), count: self.dW[indexLayer][indexNode].count)
            }
            dParametersBatchesMean.biases[indexLayer] = Array(repeating: Double(0.0), count: self.dB[indexLayer].count)
        }
        for indexBatch in 0..<self.trainingConfig.batchSize {
            for indexLayer in 0..<self.dWeightsBatch[indexBatch].count {
                for indexNode in 0..<self.dWeightsBatch[indexBatch][indexLayer].count {
                    for indexToPreviousNode in 0..<self.dWeightsBatch[indexBatch][indexLayer][indexNode].count {
                        dParametersBatchesMean.weights[indexLayer][indexNode][indexToPreviousNode] += self.dWeightsBatch[indexBatch][indexLayer][indexNode][indexToPreviousNode]
                    }
                    dParametersBatchesMean.biases[indexLayer][indexNode] += self.dBiasesBatch[indexBatch][indexLayer][indexNode]
                }
            }
        }
        let batchSizeInDouble = Double(self.trainingConfig.batchSize)
        for indexLayer in 0..<dParametersBatchesMean.weights.count {
            for indexNode in 0..<dParametersBatchesMean.weights[indexLayer].count {
                for indexToPreviousNode in 0..<dParametersBatchesMean.weights[indexLayer][indexNode].count {
                    dParametersBatchesMean.weights[indexLayer][indexNode][indexToPreviousNode] /= batchSizeInDouble
                }
                dParametersBatchesMean.biases[indexLayer][indexNode] /= batchSizeInDouble
            }
        }
        self.updateParameters(withGradients: dParametersBatchesMean)
    }

    func setLabelsMeaning(use meanings: [String]) {
        if meanings.count != self.outputSize {
            fatalError("Error: Set labels' meaning. Invalid meanings")
        }
        self.labelsMeaning = meanings
    }

    func getPredictionIndex() -> Int {
        let index: Int = self.outputLayer.valueNormalized.firstIndex(of: self.outputLayer.valueNormalized.max() ?? 0) ?? -1
        return index
    }

    func getPredictionMeaning() -> String {
        let predictionIndex = self.getPredictionIndex()
        guard let labelsMeaning = self.labelsMeaning else {
            print("Error: Get prediction meaning. Set the meanings for the labels first")
            return "Label index: \(predictionIndex)"
        }
        if predictionIndex == -1 {
            print("Error: Get prediction meaning. Prediction is empty")
            return "No Prediction"
        }
        return labelsMeaning[predictionIndex]
    }

    func getLoss() -> Double {
        guard let lossValue = self.lossLast else {
            print("Haven't propagate yet")
            return -1.0
        }
        return lossValue
    }

    func getLabelsLast() -> [Double] {
        guard let labelsLast = self.labelsLast else {
            print("Error: Print Parameters. Invalid last labels")
            return []
        }
        return labelsLast
    }

    func getProbability() -> Double {
        return self.outputLayer.valueNormalized[self.getPredictionIndex()] / 1.0
    }

    func printResults() {
        print(String(format: "Results: \(self.getPredictionMeaning()) (%.2f%%)", self.getProbability() * 100.0))
    }

    func printResultsInDetail() {
        print(String(format: "Results: \n  Predictions:\t%@", self.outputLayer.valueNormalized.map { String(format: "%.4f", $0) }.joined(separator: "\t")))
        print(String(format: "  Labels     :\t%@", self.getLabelsLast().map { String(format: "%.4f", $0) }.joined(separator: "\t")))
        print(String(format: "  Loss: %.4f", self.getLoss()))
    }

    func printParametersInDetail() {
        print("\nParameters:")
        print(String(format: "  Batch Size: %d    Learning Rate: %.4f", self.trainingConfig.batchSize, self.trainingConfig.learningRate))
        print("  Layer Structure: \(self.layerStructure)")
        var dParameters: NN.NNParameter
        if self.dParametersMean != nil {
            dParameters = self.dParametersMean!
        } else {
            dParameters = NN.NNParameter(weights: self.dW, biases: self.dB)
        }
        for indexLayer in 0..<self.layerStructure.count-1 {
            print("  Layer\(indexLayer+1):")
            print(String(format: "    Biases:\n\t%@", self.biases[indexLayer].map { String(format: "%+.4f", $0)}.joined(separator: "\t")))
            print("    Weights:")
            for indexNode in 0..<self.layerStructure[indexLayer+1] {
                print(String(format: "\t%@", self.weights[indexLayer][indexNode].map { String(format: "%+.4f", $0)}.joined(separator: "\t")))
            }
            print(String(format: "    dBiases:\n\t%@", dParameters.biases[indexLayer].map { String(format: "%+.4f", $0)}.joined(separator: "\t")))
            print("    dWeights:")
            for indexNode in 0..<self.layerStructure[indexLayer+1] {
                print(String(format: "\t%@", dParameters.weights[indexLayer][indexNode].map { String(format: "%+.4f", $0)}.joined(separator: "\t")))
            }
        }
        self.printResultsInDetail()
    }

    func printParametersByLayerInDetail() {
        print("\nParameters:")
        for (indexLayer, _) in self.weights.enumerated() {
            print("Layer \(indexLayer + 1)")
            for (indexNode, _) in self.weights[indexLayer].enumerated() {
                print(String(format: "  Node%d:\n    bias: %.3f\tdBias: %.3f\tzValue: %.3f\tactivation: %.3f", indexNode + 1, self.biases[indexLayer][indexNode], self.dB[indexLayer][indexNode], self.zs[indexLayer][indexNode], self.activations[indexLayer][indexNode]))
                print(String(format: "    Weights: %@\tdWeights: %@", self.weights[indexLayer][indexNode].map { String(format: "%.3f", $0)}.joined(separator: "\t"), self.dW[indexLayer][indexNode].map { String(format: "%.3f", $0)}.joined(separator: "\t")))
            }
        }
        self.printResultsInDetail()
    }

    //保存最佳参数
    private func saveBest() {
    }

    //恢复最佳参数
    private func loadBest() {
    }
}
