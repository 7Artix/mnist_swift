import Foundation

enum PollingMethod {
    case max
    case average
    case l2
}

struct CNNLayer {
    var filter: [[Double]]
    var poolingMethod: PollingMethod
    var poolingHeight: Int
    var poolingWidth: Int
}

class CNNModule {
    var layersCNN: [CNNLayer]
    var heightImageInputCNN: Int
    var widthImageInputCNN: Int
    var outputSizeCNN: Int

    init<T: BinaryInteger & FixedWidthInteger>(imageSample: [[T]], layersCNN: [CNNLayer]) {
        self.layersCNN = layersCNN
        self.heightImageInputCNN = imageSample.count
        self.widthImageInputCNN = imageSample.first?.count ?? 0
        self.outputSizeCNN = 0
        self.outputSizeCNN = self.fpCNN(image: imageSample, layersCNN: layersCNN).count
    }

    func fpCNN<T: BinaryInteger & FixedWidthInteger>(image: [[T]], layersCNN: [CNNLayer]) -> [Double] {
        var imageOutput = image
        for layer in layersCNN {
            imageOutput = convolution(imageOutput, with: layer.filter)
            imageOutput = pooling(imageOutput, with: layer.poolingMethod, windowHeight: layer.poolingHeight, windowWidth: layer.poolingWidth)
        }
        return imageOutput.flatMap { row in
            row.map { Double($0) }
        }
    }

    private func normalization2Dim<T_origin: BinaryInteger,T: BinaryInteger & FixedWidthInteger>(_ origin: [[T_origin]]) -> [[T]] {
        let height = origin.count
        let width = origin.first?.count ?? 0
        var normalized = Array(repeating: Array(repeating: T(0), count: width), count: height)
        if let maxValue = origin.flatMap({$0}).max() {
            if maxValue <= T.max {
                normalized = origin.map { row in 
                    row.map { T($0) }
                }
            } else {
                normalized = origin.map { row in 
                    row.map { T(round(Double($0) / Double(maxValue) * Double(T.max)))}
                }
            }
        }
        return normalized
    }
    
    //MARK: 卷积函数
    private func convolution<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]], with filter: [[Double]]) -> [[T]]{
        let filterHeight = filter.count
        let filterWidth = filter.first?.count ?? 0
        let peddingNumberWidth = Int(floor(Double(filterWidth) / 2 + 0.1))
        let peddingNumberHeight = Int(floor(Double(filterHeight) / 2 + 0.1))
        //增加Padding行列的拓展图像
        var convolutionImage: [[Int64]] = image.map { row in 
            row.map { Int64($0) }
        }
        let imageHeight = convolutionImage.count
        let imageWidth = convolutionImage.first?.count ?? 0
        var outputImageInt64 = Array(repeating: Array(repeating: Int64(0), count: imageWidth), count: imageHeight)
        //与输入图像尺寸相同的图像用以返回
        var outputImage = Array(repeating: Array(repeating: T(0), count: imageWidth), count: imageHeight)
        //添加Padding行   
        for _ in 0..<peddingNumberHeight {
            convolutionImage.insert(convolutionImage[0], at: 0)
            convolutionImage.append(convolutionImage.last ?? convolutionImage[0])
        }
        //添加Padding列
        for rowIndex in 0..<imageHeight {
            for _ in 0..<peddingNumberWidth {
                convolutionImage[rowIndex].insert(convolutionImage[rowIndex][0], at: 0)
                convolutionImage[rowIndex].append(convolutionImage[rowIndex].last ?? convolutionImage[rowIndex][0])
            }
        }
        //将四个角补充的元素取平均值均匀处理
        let rangeLeft = stride(from: peddingNumberWidth - 1, through: 0, by: -1)
        let rangeRight = (imageWidth - peddingNumberWidth)..<(imageWidth)
        let rangeTop = stride(from: peddingNumberHeight - 1, through: 0, by: -1)
        let rangeBottom = (imageHeight - peddingNumberHeight)..<imageHeight
        //处理左上角元素
        for rowIndex in rangeTop {
            for colIndex in rangeLeft {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex + 1]) +
                    Double(convolutionImage[rowIndex + 1][colIndex])) / 2))
            }
        }
        //处理左下角元素
        for rowIndex in rangeBottom {
            for colIndex in rangeLeft {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex + 1]) + 
                    Double(convolutionImage[rowIndex - 1][colIndex])) / 2))
            }
        }
        //处理右上角元素
        for rowIndex in rangeTop {
            for colIndex in rangeRight {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex - 1]) + 
                    Double(convolutionImage[rowIndex + 1][colIndex])) / 2))
            }
        }
        //处理右下角元素
        for rowIndex in rangeBottom {
            for colIndex in rangeRight {
                convolutionImage[rowIndex][colIndex] = Int64(round(
                    (Double(convolutionImage[rowIndex][colIndex - 1]) + 
                    Double(convolutionImage[rowIndex - 1][colIndex])) / 2))
            }
        }
        //卷积
        for i in peddingNumberHeight..<(peddingNumberHeight + imageHeight) {
            for j in peddingNumberWidth..<(peddingNumberWidth + imageWidth) {
                var pixelConvolutionalValue = Int64(0)
                for k in 0..<filterHeight {
                    for l in 0..<filterWidth {
                        pixelConvolutionalValue += Int64(filter[k][l]) * convolutionImage[i-peddingNumberHeight+k][j-peddingNumberWidth+l]
                    }
                }
                outputImageInt64[i][j] = pixelConvolutionalValue
            }
        }
        outputImage = normalization2Dim(outputImageInt64)
        return outputImage
    }

    //MARK: 池化函数:修剪与池化窗口倍数不匹配的像素
    private func pooling<T: BinaryInteger & FixedWidthInteger>(_ image: [[T]], with method: PollingMethod, windowHeight heightWindow: Int, windowWidth widthWindow: Int) -> [[T]] {
        let heightInput = image.count
        let widthInput = image.first?.count ?? 0
        let heightOutput = heightInput / heightWindow
        let widthOutput = widthInput / widthWindow
        var imageOutput = Array(repeating: Array(repeating: T(0), count: widthOutput), count: heightOutput)
        for indexRowOutput in 0..<heightOutput {
            for indexColOutput in 0..<widthOutput {
                var poolingValuesTemp: [T] = []
                for indexRowWindow in 0..<heightWindow {
                    for indexColWindow in 0..<widthWindow {
                        poolingValuesTemp.append(image[indexRowOutput*heightWindow+indexRowWindow][indexColOutput*widthWindow+indexColWindow])
                    }
                }
                imageOutput[indexRowOutput][indexColOutput] = poolingAction(poolingValuesTemp, with: method)
            }
        }
        return imageOutput
    }
    private func poolingAction<T: BinaryInteger & FixedWidthInteger>(_ values: [T], with method: PollingMethod) -> T {
        var result: T
        switch method {
            case .average:
                var valueSum: Int64 = 0
                valueSum = Int64(values.reduce(0, {$0 + $1}))
                result = T(valueSum / Int64(values.count))
            case .l2:
                var squareSum: UInt64 = 0
                squareSum = UInt64(values.reduce(0, {$0 + $1 * $1}))
                result = T(sqrt(Double(squareSum)))
            case .max:
                result = values.max() ?? T(0)
        }
        return result
    }
}

class CNN: NN {
    let moduleCNN: CNNModule
    init(networkConfig: NNConfig, trainingConfig: TrainingConfig, moduleCNN: CNNModule) {
        self.moduleCNN = moduleCNN
        let networkConfigCNN = NNConfig(inputSize: moduleCNN.outputSizeCNN, structure: networkConfig.structure, outputLayer: networkConfig.outputLayer)
        super.init(networkConfig: networkConfigCNN, trainingConfig: trainingConfig)
    }
}