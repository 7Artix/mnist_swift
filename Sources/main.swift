import Foundation
import CoreGraphics

let pathTrainingImages = "./mnist_database/train-images.idx3-ubyte"
let pathTrainingLabels = "./mnist_database/train-labels.idx1-ubyte"
let pathTestImages = "./mnist_database/t10k-images.idx3-ubyte"
let pathTestLabels = "./mnist_database/t10k-labels.idx1-ubyte"
let trainingData = Database(imagesPath: pathTrainingImages, labelsPath: pathTrainingLabels)
let testData = Database(imagesPath: pathTestImages, labelsPath: pathTestLabels)

let mnistTraining = MNIST(database: trainingData)
let mnistTest = MNIST(database: testData)

// let firstImage = mnistTraining.getImage(index: 0).0
// let firstLabel = mnistTraining.getImage(index: 0).1

// do {
//     let imageRecipeGray8 = ImageRecipe(width: 28, height: 28, pixelFormat: .GRAY8, provider: try .makeProviderGrayscale(from: firstImage))
//     let cgImageGray8 = try CGImage.createCGImage(imageRecipe: imageRecipeGray8)
//     try cgImageGray8.saveCGImage(useName: "first_image_label_\(firstLabel)", toPath: "./test", as: .png)
// }

// var networkStructure: [[NodeStructure]] = []
// networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 4))
// networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 3))
// networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 2))
// var outputlayer = OutputLayer(outputSize: 2, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
// var networkConfig = NNConfig(inputSize: 5, structure: networkStructure, outputLayer: outputlayer)
// var trainingConfig = TrainingConfig(batchSize: 1, learningRate: 0.005, negativeAttempts: 5)
// let network = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
// print("Layers structure\(network.layerStructure)")
// print("Weights structure: \(network.weightStructure)")
// network.printParameters()
// network.fp(input: [1.0, 1.0, 2.0, 3.0, 5.0], labels: [1.0, 0.0])
// print("\nAfter FP:")
// network.printParameters()
// network.bp()
// var oneStepParameter: NN.NNParameter = NN.NNParameter(weights: network.dWeights, biases: network.dBiases)
// network.updateParameters(withGradients: oneStepParameter)
// print("\nAfter Descent:")
// network.printParameters()

let inputData = [1.0, 2.0, 3.0, 4.0]
let label = [1.0, 0.0]


var networkStructure: [[NodeStructure]] = []
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 4))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 5))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 6))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 2))
var outputlayer = OutputLayer(outputSize: 2, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
var networkConfig = NNConfig(inputSize: 4, structure: networkStructure, outputLayer: outputlayer)
var trainingConfig = TrainingConfig(batchSize: 10, learningRate: 0.05, negativeAttempts: 5)
let network2 = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
network2.fp(input: inputData, labels: label)
network2.printResults()
network2.bp()
network2.descentSingleStep()
network2.printParametersInDetail()
network2.fp(input: inputData, labels: label)
network2.printResults()

// let startTime = Date()
// for i in 0..<100 {
//     print("\rProcessing: \(i + 1)/100", terminator: "")
//     fflush(stdout) // 刷新标准输出
//     network.fp(input: testImage1Dim, labels: testLabel)
// }
// let endTime = Date()
// print("\nDone! in \(endTime.timeIntervalSince(startTime))")

