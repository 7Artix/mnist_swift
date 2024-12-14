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
    //输出层(激活值): 神经网络输出的激活值
    var valueNetwork: [Double]
    //输出层(归一值): 归一化后值
    var valueNormalized: [Double]
    //对损失函数求导梯度
    var dLoss: [Double]?
    //对归一化函数求导梯度
    var dNormalization: [Double]?
    //归一化函数
    var normalizationFunction: NormalizationFunction
    //损失函数
    var lossFunction: LossFunction

    init(outputSize: Int, normalizationFunction: NormalizationFunction, lossFunction: LossFunction) {
        self.outputSize = outputSize
        self.valueNetwork = Array(repeating: Double(0.0), count: outputSize)
        self.valueNormalized = Array(repeating: Double(0.0), count: outputSize)
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
    //激活值矩阵
    var activations: [[Double]]
    //权重矩阵
    var biases: [[Double]]
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

    init(networkConfig: NNConfig, trainingConfig: TrainingConfig){
        self.trainingConfig = trainingConfig
        self.outputLayer = networkConfig.outputLayer
        self.inputSize = networkConfig.inputSize
        self.outputSize = self.outputLayer.outputSize
        //从网络结构获取层结构
        layerStructure = networkConfig.structure.map {$0.count}
        //层结构加入输入层数量, 但并无实际节点, 仅便于后续建立连接赋权重
        layerStructure.insert(networkConfig.inputSize, at: 0)
        //层结构加入输出层数量, 实际节点由outputLayer, 仅便于后续建立连接赋权重
        layerStructure.append(outputSize)
        //建立权重结构, 多加入了从输出层到归一化层的无效权重
        weightStructure = []
        for index in 0..<(layerStructure.count - 1) {
            let nodeCount = layerStructure[index]
            let nextLayerNodeCount = layerStructure[index + 1]
            weightStructure.append(Array(repeating: nodeCount, count: nextLayerNodeCount))
        }
        //去除输出层到归一化层的无效权重
        weightStructure = weightStructure.dropLast()
        //初始化
        activations = layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0) }
        biases = layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        activationFunctions = layerStructure.dropFirst().map { Array(repeating: ReLU(), count: $0)}
        dBiases = layerStructure.dropFirst().map { Array(repeating: Double(0.0), count: $0)}
        weights = weightStructure.map { node in node.map {Array(repeating: 0.0, count: $0)}}
        dWeights = weightStructure.map { node in node.map {Array(repeating: 0.0, count: $0)}}
        //根据配置初始化权重值, 偏置值与激活函数
        for (indexLayer, layer) in networkConfig.structure.enumerated() {
            let inputNodeCount = layerStructure[indexLayer]
            let outputNodeCount = layerStructure[indexLayer + 2]
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
    }

    func bp(input: [Double], labels: [Double]) {
        
    }

    func descent(input: [Double], labels: [Double]) {

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

    //保存最佳参数
    private func saveBest() {
    }

    //恢复最佳参数
    private func loadBest() {
    }
}
