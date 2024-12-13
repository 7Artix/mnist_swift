import Foundation

//MARK: 激活函数
protocol ActivationFunction {
    static func forward(_ x: Double) -> Double
    static func backward(_ x: Double) -> Double
}
//ReLU
class ReLU: ActivationFunction {
    static func forward(_ x: Double) -> Double {
        return max(0, x)
    }
    static func backward(_ x: Double) -> Double {
        return x > 0 ? 1.0 : 0.0
    }
}
//Sigmoid
class Sigmoid: ActivationFunction {
    static func forward(_ x: Double) -> Double {
        return 1 / (1 + exp(-x))
    }
    static func backward(_ x: Double) -> Double {
        let y = forward(x)
        return y * (1 - y)
    }
}

//MARK: 权重初始化函数
//Glorot初始化 for weight
func glorotInitializer(inputSize: Int?, outputSize: Int?) -> Double {
    let limit = sqrt(6.0 / Double((inputSize ?? 1) + (outputSize ?? 1)))
    return Double.random(in: -limit...limit)
}
//HE初始化 for weight
func heInitializer(inputSize: Int?, _: Int?) -> Double {
    let stdDev = sqrt(2.0 / Double((inputSize ?? 1)))
    return generateNormal(mean: 0.0, stdDev: stdDev)
}
//常值初始化 for weight
func constantInitializer(_: Int?, _: Int?) -> Double {
    return 0.5
}

//MARK: 损失函数
protocol LossFunction {
    init()
    func forward(predictions: [Double], labels: [Double]) -> Double
    func backward(predictions: [Double], labels: [Double]) -> [Double]
}
// 交叉熵损失函数
class CrossEntropy: LossFunction {
    required init() {}
    func forward(predictions: [Double], labels: [Double]) -> Double {
        guard predictions.count == labels.count else {
            fatalError("The number of predictions doesn't match the labels'")
        }
        var loss: Double = 0.0
        for (index, label) in labels.enumerated() {
            loss -= label * log(predictions[index] + 1e-15) // 防止log(0)的情况
        }
        return loss / Double(labels.count)
    }
    func backward(predictions: [Double], labels: [Double]) -> [Double] {
        guard predictions.count == labels.count else {
            fatalError("The number of predictions doesn't match the labels'")
        }
        return zip(predictions, labels).map { (prediction, label) in
            prediction - label
        }
    }
}

//MARK: 归一化函数
protocol NormalizationFunction {
    func forward(inputAll: [Double], forNode: Int) -> Double
    func backward(dInputAll: [Double], forNode: Int) -> Double
}
//softmax
class Softmax: NormalizationFunction {
    func forward(inputAll: [Double], forNode index: Int) -> Double {
        let maxInput = inputAll.max() ?? 0.0
        //减去最大值, 防止由于指数过大导致溢出, 或指数过小导致精度问题
        let expValues = inputAll.map { exp($0 - maxInput) }
        let sumExp = expValues.reduce(0, +)
        return expValues[index] / sumExp
    }
    //softmax求导为全梯度
    func backward(dInputAll: [Double], forNode index: Int) -> Double {
        var dValue: Double = 0.0
        for i in 0..<dInputAll.count {
            if i == index {
                dValue += dInputAll[i] * (1 - dInputAll[i])
            }else {
                dValue += dInputAll[i] * dInputAll[index]
            }
        }
        return dValue
    }
}

//MARK: 生成函数
//正态分布生成
func generateNormal(mean: Double, stdDev: Double) -> Double {
    let u1 = Double.random(in: 0...1)
    let u2 = Double.random(in: 0...1)
    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    return mean + z * stdDev
}
