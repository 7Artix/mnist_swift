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
    var valueNetwork: [Double]
    var valueNormalized: [Double]
    var dLoss: [Double]?
    var dNormalization: [Double]?
    var normalizationFunction: NormalizationFunction
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
    let inputSize: Int
    var structure: [[NodeStructure]]
    var outputLayer: OutputLayer
    init(inputSize: Int, structure: [[NodeStructure]], outputLayer: OutputLayer) {
        guard let outputSizeFromStructure = structure.last?.count else {
            fatalError("Error with input node structure")
        }
        if (outputLayer.outputSize != outputSizeFromStructure) {
            fatalError("Output layer doesn't match the network")
        }
        self.inputSize = inputSize
        self.structure = structure
        self.outputLayer = outputLayer
    }
}

class NN {
    struct NNParameter {
        var weights: [[[Double]]]
        var biases: [[Double]]
    }
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
    //历史最佳参数
    var historyBest: NNParameter?

    init(config: NNConfig){
        self.outputLayer = config.outputLayer
        self.outputSize = self.outputLayer.outputSize
        //从网络结构获取层结构
        layerStructure = config.structure.map {$0.count}
        //层结构加入输入层数量, 但并无实际节点, 仅便于后续建立连接赋权重
        layerStructure.insert(config.inputSize, at: 0)
        //层结构加入输出层数量, 实际节点由outputLayer, 仅便于后续建立连接赋权重
        layerStructure.append(outputSize)
        //建立权重结构, 多加入了从输出层到归一化层的无效权重
        weightStructure = []
        for (index, nodeCount) in layerStructure.enumerated() {
            weightStructure.append( Array(repeating: nodeCount, count: layerStructure[index + 1]) )
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
        for (indexLayer, layer) in config.structure.enumerated() {
            let inputNodeCount = layerStructure[indexLayer]
            let outputNodeCount = layerStructure[indexLayer+2]
            for (indexNode, node) in layer.enumerated() {
                biases[indexLayer][indexNode] = node.bias
                activationFunctions[indexLayer][indexNode] = node.activationFunction
                for indexWeight in 0..<weights[indexLayer][indexNode].count {
                    weights[indexLayer][indexNode][indexWeight] = node.weightInitializer(inputNodeCount, outputNodeCount)
                }
            }
        }
    }

    func fp(dataInOneDim: [Double], labels: [Double]) {
        
    }

    //保存最佳参数
    private func saveBest() {
    }

    //恢复最佳参数
    private func loadBest() {
    }
}
