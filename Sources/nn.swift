import Foundation

struct NodeStructure {
    var activationFunction: ActivationFunction
    var weightInitializer: (Int?, Int?) -> Double
    var bias: Double
}

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
        var weights: [[[Double]]] // 按层存储每个节点的权重
        var biases: [[Double]]    // 按层存储每个节点的偏置
    }
    var weights: [[[Double]]]
    var activations: [[Double]]
    var biases: [[Double]]
    var outputLayer: OutputLayer
    var outputSize: Int
    var layerStructure: [Int]
    var dWeights: [[[Double]]]
    var dBiases: [[Double]]
    var historyBest: NNParameter?

    init(config: NNConfig){
        self.outputLayer = config.outputLayer
        self.outputSize = self.outputLayer.outputSize
        layerStructure = config.structure.map {$0.count}
        layerStructure.insert(config.inputSize, at: 0)
        for (indexLayer, layer) in config.structure.enumerated() {
            for (indexNode, node) in layer.enumerated() {
                activations[indexLayer][indexNode] = 0.0
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
