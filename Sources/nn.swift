import Foundation

struct NodeConfig {
    let inputSize: Int
    let outputSize: Int?
    let initBias: Double
    let weightInitializer: (Int, Int?) -> Double
    let biasInitializer: (Double) -> Double
}

struct LayerConfig {
    let nodeCount: Int
    let inputSize: Int
    let outputSize: Int?
    let initBiasValue: Double
    let weightInitializer: (Int, Int?) -> Double
    let biasInitializer: (Double) -> Double
}

struct NNConfig {
    var layerConfigs: [LayerConfig]
    let inputSize: Int
    var outputFunction: ([Double]) -> [Double] = softmax(_:)
    var costFunction: ([Double],[Double]) -> Double = crossEntropyLoss(predictions:labels:)

    init(inputSize: Int, initBias: Double, layerStructure: [Int], weightInitializer: @escaping (Int, Int?) -> Double, biasInitializer: @escaping (Double) -> Double) {
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
    var weight: [Double]
    var inputSize: Int
    var outputSize: Int?

    //MARK: 初始化函数
    //Glorot初始化 for weight
    static func glorotInitializer(inputSize: Int, outputSize: Int?) -> Double {
        let limit = sqrt(6.0 / Double(inputSize + (outputSize ?? 0)))
        return Double.random(in: -limit...limit)
    }
    //HE初始化 for weight
    static func heInitializer(inputSize: Int, _: Int?) -> Double {
        let stdDev = sqrt(2.0 / Double(inputSize))
        return generateNormal(mean: 0.0, stdDev: stdDev)
    }
    //常值初始化 for weight
    static func constantInitializer(_: Int, _: Int?) -> Double {
        return 0.5
    }
    //bias初始化
    static func biasInitializer(value: Double) -> Double {
        return value
    }

    init(config: NodeConfig) {
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        self.bias = config.biasInitializer(config.initBias);
        self.weight = (0..<inputSize).map { _ in config.weightInitializer(config.inputSize, config.outputSize)}
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
    var layers: [Layer]
    var historyBest: BestParameter?
    let layerCount: Int

    init(config: NNConfig){
        self.layers = config.layerConfigs.map { Layer(config: $0) }
        self.layerCount = self.layers.count
    }

    func fp() {
        
    }

    //保存最佳参数
    private func saveBest() {
        let weights = layers.map { layer in
            layer.nodes.map { node in node.weight }
        }
        let biases = layers.map { layer in
            layer.nodes.map { node in node.bias }
        }
        self.historyBest = BestParameter(nodeWeights: weights, nodeBiases: biases)
    }

    //恢复最佳参数
    private func loadBest() {
        guard let historyBest = historyBest else { print("Without best record to load"); return}
        for (layerIndex, layer) in layers.enumerated() {
            for (nodeIndex, node) in layer.nodes.enumerated() {
                node.weight = historyBest.nodeWeights[layerIndex][nodeIndex]
                node.bias = historyBest.nodeBiases[layerIndex][nodeIndex]
            }
        }
    }
}
