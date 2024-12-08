import Foundation

//MARK: 激活函数
func relu(_ x: Double) -> Double {
    return max(0, x)
}

func sigmoid(_ x: Double) -> Double {
    return 1 / (1 + exp(-x))
}

//MARK: 损失函数

//MARK: 归一化

//MARK: 生成函数
//正态分布生成
func generateNormal(mean: Double, stdDev: Double) -> Double {
    let u1 = Double.random(in: 0...1)
    let u2 = Double.random(in: 0...1)
    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
    return mean + z * stdDev
}

//MARK: 初始化函数
//Glorot初始化 for weight
func glorotInitializer(inputSize: Int, outputSize: Int?) -> Double {
    let limit = sqrt(6.0 / Double(inputSize + (outputSize ?? 0)))
    return Double.random(in: -limit...limit)
}
//HE初始化 for weight
func heInitializer(inputSize: Int, _: Int?) -> Double {
    let stdDev = sqrt(2.0 / Double(inputSize))
    return generateNormal(mean: 0.0, stdDev: stdDev)
}
//常值初始化 for weight
func constantInitializer(_: Int, _: Int?) -> Double {
    return 0.5
}
//bias初始化
func biasInitializer(value: Double) -> Double {
    return value
}