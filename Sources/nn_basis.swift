import Foundation

//MARK: 激活函数
protocol ActivationFunction {
    func forward(_ x: Double) -> Double
    func backward(_ x: Double) -> Double
}

class ReLU: ActivationFunction {
    func forward(_ x: Double) -> Double {
        return max(0, x)
    }
    func backward(_ x: Double) -> Double {
        return x > 0 ? 1.0 : 0.0
    }
}

class Sigmoid: ActivationFunction {
    func forward(_ x: Double) -> Double {
        return 1 / (1 + exp(-x))
    }
    func backward(_ x: Double) -> Double {
        let y = forward(x)
        return y * (1 - y)
    }
}

//MARK: 损失函数
protocol LossFunction {
    func forward(predictions: [Double], labels: [Double]) -> Double
    func backward(predictions: [Double], labels: [Double]) -> [Double]
}
// 交叉熵损失函数
func crossEntropyLoss(predictions: [Double], labels: [Double]) -> Double {
    // 确保 predictions 和 labels 长度一致
    guard predictions.count == labels.count else {
        fatalError("Predictions and labels must have the same length.")
    }
    var loss: Double = 0.0
    for (index, label) in labels.enumerated() {
        loss -= label * log(predictions[index] + 1e-15) // 防止log(0)的情况
    }
    return loss / Double(labels.count)
}

//MARK: 归一化函数
//softmax
func softmax(_ input: [Double]) -> [Double] {
    let maxInput = input.max() ?? 0.0
    //减去最大值, 防止由于指数过大导致溢出, 或指数过小导致精度问题
    let expValues = input.map { exp($0 - maxInput) }
    let sumExp = expValues.reduce(0, +)
    return expValues.map { $0 / sumExp }
}

//MARK: 生成函数
//正态分布生成
func generateNormal(mean: Double, stdDev: Double) -> Double {
    let u1 = Double.random(in: 0...1)
    let u2 = Double.random(in: 0...1)
    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    return mean + z * stdDev
}
