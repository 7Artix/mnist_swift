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