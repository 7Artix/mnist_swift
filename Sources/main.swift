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

var networkStructure: [[NodeStructure]] = []
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 512))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 256))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 128))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 64))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 10))
var outputlayer = OutputLayer(outputSize: 10, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
var networkConfig = NNConfig(inputSize: 784, structure: networkStructure, outputLayer: outputlayer)
var trainingConfig = TrainingConfig(batchSize: 10, learningRate: 0.001, negativeAttempts: 5)
let network = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)

// let testImage1Dim = mnistTraining.getImage(index: 0).0.flatMap{ $0 }.map { Double($0) }
// let testLabel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

// network.fp(input: testImage1Dim, labels: testLabel)
// print(network.outputLayer.valueNetwork)
// print(network.outputLayer.valueNormalized)
// print(network.lastLoss ?? "Call fp first")

let testImagesBatch = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 10).0
let testLabelsBatch = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 10).1

network.descentbatches(inputs: testImagesBatch, labels: testLabelsBatch)

let checkImage = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 1).0[0]
let checkLabel = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 1).1[0]

network.fp(input: checkImage, labels: checkLabel)
print(network.getLoss())

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
