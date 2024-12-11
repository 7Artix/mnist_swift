import Foundation

struct NNConfig {
    let inputSize: Int
    var outputFunction: ([Double]) -> [Double] = softmax(_:)
    var lossFunctionType: LossFunction.Type = CrossEntropy.self
    lazy var lossFunction: LossFunction = lossFunctionType.init()
    
    struct NodeStructure {
        var activationFunction: ActivationFunction
        var weightInitializer: (Int?, Int?) -> Double
        var bias: Double
    }
    struct NormNodeStructure {
        var valueMe: Double
        var valueAll: [Double]
        var 
    }

    init(inputSize: Int, structure: [[NodeStructure]], 
    weightInitializer: @escaping (Int, Int?) -> Double, biasInitializer: @escaping (Double) -> Double) {
        self.inputSize = inputSize
        var previousNodeCount = inputSize
        self.layerConfigs = layerStructure.enumerated().map { (index, nodeCount) in
            let nextNodeCount = index + 1 < layerStructure.count ? layerStructure[index + 1] : nil
            let layerConfig = LayerConfig(nodeCount: nodeCount, inputSize: previousNodeCount, outputSize: nextNodeCount, initBiasValue: initBias, weightInitializer: weightInitializer, biasInitializer: biasInitializer)
            previousNodeCount = nodeCount
            return layerConfig
        }
    }
}

class Node {
    var activation: Double = 0.0
    var bias: Double
    var inputSize: Int
    var outputSize: Int?

    init(config: NodeConfig) {
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        self.bias = config.biasInitializer(config.initBias);
    }
}

class Layer {
    var nodes: [Node]
    var nodeCount: Int
    var inputSize: Int
    var outputSize: Int?

    init(config: LayerConfig) {
        self.nodeCount = config.nodeCount
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        let nodeConfig = NodeConfig(inputSize: config.inputSize, outputSize: config.outputSize, initBias: config.initBiasValue, weightInitializer: config.weightInitializer, biasInitializer: config.biasInitializer)
        self.nodes = (0..<nodeCount).map {_ in Node(config: nodeConfig)}
    }
}

class NN {
    struct BestParameter {
        var nodeWeights: [[[Double]]] // 按层存储每个节点的权重
        var nodeBiases: [[Double]]    // 按层存储每个节点的偏置
    }
    var weights: [[[Double]]]
    var activations: [[Double]]
    var historyBest: BestParameter?

    init(config: NNConfig){
        self.layers = config.layerConfigs.map { Layer(config: $0) }
        self.layerCount = self.layers.count
    }

    func fp(dataInOneDim: [Double], labels: [Double]) {
        
    }

    //保存最佳参数
    private func saveBest() {
        let biases = layers.map { layer in
            layer.nodes.map { node in node.bias }
        }
        self.historyBest = BestParameter(nodeWeights: weights, nodeBiases: biases)
    }

    //恢复最佳参数
    private func loadBest() {
        guard let historyBest = historyBest else { print("Without best record to load"); return}
        weights = historyBest.nodeWeights
        for (layerIndex, layer) in layers.enumerated() {
            for (nodeIndex, node) in layer.nodes.enumerated() {
                node.bias = historyBest.nodeBiases[layerIndex][nodeIndex]
            }
        }
    }
}
