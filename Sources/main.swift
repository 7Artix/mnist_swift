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

// var networkStructure: [[NodeStructure]] = []
// networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 3))
// networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 3))
// networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 2))
// var outputlayer = OutputLayer(outputSize: 2, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
// var networkConfig = NNConfig(inputSize: 4, structure: networkStructure, outputLayer: outputlayer)
// var trainingConfig = TrainingConfig(batchSize: 1, learningRate: 0.01, negativeAttempts: 5)
// let network = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
// for i in 0...3 {
//     network.weights[0][0][i] = Double(i+1) * 0.1
//     network.weights[0][1][i] = Double(i+1) * 0.1 + 0.4
//     network.weights[0][2][i] = Double(i+1) * 0.1 + 0.8
// }
// for i in 0...2 {
//     network.weights[1][0][i] = Double(i+1) * 0.1
//     network.weights[1][1][i] = Double(i+1) * 0.1 + 0.3
//     network.weights[1][2][i] = Double(i+1) * 0.1 + 0.6

//     network.weights[2][0][i] = Double(i+1) * 0.1
//     network.weights[2][1][i] = Double(i+1) * 0.1 + 0.3
// }
// network.fp(input: inputData, labels: label)
// network.printParameters()
// network.bp()
// network.printParameters()


var networkStructure: [[NodeStructure]] = []
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 3))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 3))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 2))
var outputlayer = OutputLayer(outputSize: 2, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
var networkConfig = NNConfig(inputSize: 4, structure: networkStructure, outputLayer: outputlayer)
var trainingConfig = TrainingConfig(batchSize: 1, learningRate: 0.01, negativeAttempts: 5)
let network2 = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
network2.weights[0][0][0] = 0.2
network2.weights[0][0][1] = -0.3
network2.weights[0][0][2] = 0.5
network2.weights[0][0][3] = 0.1
network2.weights[0][1][0] = -0.4
network2.weights[0][1][1] = 0.6
network2.weights[0][1][2] = -0.1
network2.weights[0][1][3] = 0.2
network2.weights[0][2][0] = 0.3
network2.weights[0][2][1] = -0.2
network2.weights[0][2][2] = 0.4
network2.weights[0][2][3] = -0.5

network2.weights[1][0][0] = 0.5
network2.weights[1][0][1] = -0.4
network2.weights[1][0][2] = 0.3
network2.weights[1][1][0] = -0.3
network2.weights[1][1][1] = 0.7
network2.weights[1][1][2] = -0.2
network2.weights[1][2][0] = 0.6
network2.weights[1][2][1] = -0.1
network2.weights[1][2][2] = 0.4

network2.weights[2][0][0] = 0.4
network2.weights[2][0][1] = -0.5
network2.weights[2][0][2] = 0.6
network2.weights[2][1][0] = -0.7
network2.weights[2][1][1] = 0.8
network2.weights[2][1][2] = -0.9
network2.fp(input: inputData, labels: label)
network2.printParametersByLayer()
print(network2.weightStructure)
print(network2.layerStructure)
network2.bp()
network2.printParametersByLayer()



// let testImage1Dim = mnistTraining.getImage(index: 0).0.flatMap{ $0 }.map { Double($0) }
// let testLabel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
// let testImagesBatch = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 10).0
// let testLabelsBatch = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 10).1
// network.descentbatches(inputs: testImagesBatch, labels: testLabelsBatch)

// network.fp(input: testImage1Dim, labels: testLabel)
// print(network.outputLayer.valueNetwork)
// print(network.outputLayer.valueNormalized)
// print(network.lastLoss ?? "Call fp first")

// let checkImage = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 1).0[0]
// let checkLabel = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 1).1[0]

// print("weights: \(network.weights[4][0])")
// print("gradients: \(network.dWeightsBatch)")
// network.descentbatches(inputs: testImagesBatch, labels: testLabelsBatch)
// network.fp(input: checkImage, labels: checkLabel)
// print(network.getLoss())
// print("weights: \(network.weights[4][0])")
// print("gradients: \(network.dWeightsBatch[0][4][0])")

// let startTime = Date()
// for i in 0..<100 {
//     print("\rProcessing: \(i + 1)/100", terminator: "")
//     fflush(stdout) // 刷新标准输出
//     network.fp(input: testImage1Dim, labels: testLabel)
// }
// let endTime = Date()
// print("\nDone! in \(endTime.timeIntervalSince(startTime))")

// print(network.activations)
// print(network.activationFunctions)
// print(network.biases)
// print(network.dBiases)
// print(network.dBiasesBatch)
// print(network.weights)
// print(network.dWeights)
// print(network.dWeightsBatch)
