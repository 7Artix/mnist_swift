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
    var zValues: [[Double]]
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
    var delta: [[Double]]
    //相对权重张量的梯度
    var dWeights: [[[Double]]]
    //相对偏置矩阵的梯度
    var dBiases: [[Double]]
    //batch多样本梯度存储
    var dWeightsBatch: [[[[Double]]]] = []
    var dBiasesBatch: [[[Double]]] = []
    //历史最佳参数
    var historyBest: NNParameter?
    //上次前向传播的预测结果
    var lastPrediction: Int?
    //上次前向传播的损失值
    var lastLoss: Double?
    //上次输入值
    var InputsLast: [Double]?

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
        self.zValues = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.activations = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.activationFunctions = self.layerStructure.dropFirst().map { Array(repeating: ReLU(), count: $0)}
        self.biases = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.weights = self.weightStructure.map { node in node.map {Array(repeating: 0.0, count: $0)}}
        self.delta = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.dBiases = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.dWeights = self.weightStructure.map { node in node.map {Array(repeating: 0.0, count: $0)}}
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
        self.InputsLast = input
        var activationsPreviousLayer = input
        for(indexLayer, weightsLayer) in self.weights.enumerated() {
            let countNode = self.activations[indexLayer].count
            for indexNode in 0..<countNode {
                let weightsNode = weightsLayer[indexNode]
                let weightedSum = zip(weightsNode, activationsPreviousLayer).reduce(0.0) { sum, pair in sum + pair.0 * pair.1} + self.biases[indexLayer][indexNode]
                self.zValues[indexLayer][indexNode] = weightedSum
                self.activations[indexLayer][indexNode] = self.activationFunctions[indexLayer][indexNode].forward(weightedSum)
            }
            activationsPreviousLayer = self.activations[indexLayer]
        }
        guard let valueNetwork = zValues.last else {
            fatalError("Error: FP. Invalid output layer values")
        }
        self.outputLayer.valueNetwork = valueNetwork
        for (index, _) in self.outputLayer.valueNetwork.enumerated() {
            self.outputLayer.valueNormalized[index] = self.outputLayer.normalizationFunction.forward(inputAll: self.outputLayer.valueNetwork, indexNode: index)
        }
        self.lastLoss = self.outputLayer.lossFunction.forward(predictions: self.outputLayer.valueNormalized, labels: labels)
    }

    func bp() {
        guard let inputs = self.InputsLast else {
            fatalError("Error: BP. Call FP first")
        }
        for index in 0..<self.outputSize {
            self.outputLayer.dLoss[index] = self.outputLayer.lossFunction.backward(indexNode: index)
        }
        self.outputLayer.lossFunction.reset()
        for index in 0..<self.outputSize {
            self.outputLayer.dNormalization[index] = self.outputLayer.normalizationFunction.backward(dInputAll: self.outputLayer.dLoss, indexNode: index)
        }
        var gradientsFromPreviousLayer = self.outputLayer.dNormalization
        for indexLayer in (0..<self.weights.count).reversed() {
            //上一层时输入层时不需要计算梯度(gradientsCurrentLayerForPrevious)
            if indexLayer == 0 {

            }
            let countNode = self.activations[indexLayer].count
            let countNodeLayerPrevious = indexLayer > 0 ? self.activations[indexLayer - 1].count : 0
            var gradientsCurrentLayerForPrevious = Array(repeating: Double(0.0), count: countNodeLayerPrevious)
            for indexNode in 0..<countNode {
                let activation = self.activations[indexLayer][indexNode]
                let gradientActivation = self.activationFunctions[indexLayer][indexNode].backward(activation)
                self.dBiases[indexLayer][indexNode] = gradientsFromPreviousLayer[indexNode]
                if indexLayer == 0 {
                    for indexToPreviousNode in 0..<self.weights[indexLayer][indexNode].count {
                        self.dWeights[indexLayer][indexNode][indexToPreviousNode] = gradientsFromPreviousLayer[indexNode] * inputs[indexToPreviousNode]
                    }
                } else {
                    for indexToPreviousNode in 0..<self.weights[indexLayer][indexNode].count {
                        self.dWeights[indexLayer][indexNode][indexToPreviousNode] = gradientsFromPreviousLayer[indexNode] * self.activations[indexLayer - 1][indexToPreviousNode] * self.activationFunctions[indexLayer][indexNode].backward(self.zValues[indexLayer][indexNode])
                    }
                }
                if indexLayer > 0 {
                    gradientsCurrentLayerForPrevious[indexNode] = zip(gradientsFromPreviousLayer, self.weights[indexLayer].map { $0[indexNode] }).reduce(0) { sum, pair in sum + pair.0 * pair.1 } * gradientActivation
                }
            }
            gradientsFromPreviousLayer = gradientsCurrentLayerForPrevious
        }
        //反向传播后将输入值/激活值/线性叠加值清零
        InputsLast = nil
        self.zValues = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        self.activations = self.layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
    }

    func resetGradients() {
        for indexLayer in 0..<self.dWeights.count {
            for indexNode in 0..<self.dWeights[indexLayer].count {
                self.dWeights[indexLayer][indexNode] = Array(repeating: Double(0.0), count: self.dWeights[indexLayer][indexNode].count)
            }
            self.dBiases[indexLayer] = Array(repeating: Double(0.0), count: self.dBiases[indexLayer].count)
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
            self.dWeightsBatch.append(self.dWeights)
            self.dBiasesBatch.append(self.dBiases)
            self.resetGradients()
        }
        var dParametersBatchesMean = NNParameter(
            weights: self.dWeights.map { layer in
                layer.map { Array(repeating: Double(0.0), count: $0.count) }
            },
            biases: self.dBiases.map { layer in
                Array(repeating: Double(0.0), count: layer.count)
            }
        )
        for indexLayer in 0..<self.dWeights.count {
            for indexNode in 0..<self.dWeights[indexLayer].count {
                dParametersBatchesMean.weights[indexLayer][indexNode] = Array(repeating: Double(0.0), count: self.dWeights[indexLayer][indexNode].count)
            }
            dParametersBatchesMean.biases[indexLayer] = Array(repeating: Double(0.0), count: self.dBiases[indexLayer].count)
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

    func getPrediction() -> Int {
        guard let predictionValue = self.lastPrediction else {
            print("Haven't propagate yet")
            return -1
        }
        return predictionValue
    }

    func getLoss() -> Double {
        guard let lossValue = self.lastLoss else {
            print("Haven't propagate yet")
            return -1
        }
        return lossValue
    }

    func printParametersByLayer() {
        print("\nHere's the parameters:")
        for (indexLayer, _) in self.weights.enumerated() {
            print("Layer \(indexLayer + 1)")
            for (indexNode, _) in self.weights[indexLayer].enumerated() {
                print(String(format: "  Node: %d, bias: %.3f, dBias: %.3f, zValue: %.3f, activation: %.3f", indexNode + 1, self.biases[indexLayer][indexNode], self.dBiases[indexLayer][indexNode], self.zValues[indexLayer][indexNode], self.activations[indexLayer][indexNode]))
                print(String(format: "    Weights: %@    dWeights: %@", self.weights[indexLayer][indexNode].map { String(format: "%.3f", $0)}.joined(separator: ", "), self.dWeights[indexLayer][indexNode].map { String(format: "%.3f", $0)}.joined(separator: ", ")))
            }
        }
        print(String(format: "Output Layer: \n  valuesNetwork: %@\n  Predictions: %@", self.outputLayer.valueNetwork.map { String(format: "%.3f", $0) }.joined(separator: ", "), self.outputLayer.valueNormalized.map { String(format: "%.3f", $0) }.joined(separator: ", ")))
        print(String(format: "  dLoss:         %@", self.outputLayer.dLoss.map { String(format: "%.3f", $0) }.joined(separator: ", ")))
        print(String(format: "  dNormalizaton: %@", self.outputLayer.dNormalization.map { String(format: "%.3f", $0) }.joined(separator: ", ")))
    }

    //保存最佳参数
    private func saveBest() {
    }

    //恢复最佳参数
    private func loadBest() {
    }
}
