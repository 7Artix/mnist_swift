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

let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: 20)
let testImages = mnistTest.getImagesBatchForNetwork(fromIndex: 0, batchSize: 20)

let trainingImage0 = trainingImages.0[0]
let trainingLabel0 = trainingImages.1[0]
let trainingImage1 = trainingImages.0[1]
let trainingLabel1 = trainingImages.1[1]
let trainingImage2 = trainingImages.0[2]
let trainingLabel2 = trainingImages.1[2]
let trainingImage3 = trainingImages.0[3]
let trainingLabel3 = trainingImages.1[3]

let labelsMeaning = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

var networkStructure: [[NodeStructure]] = []
networkStructure.append(Array(repeating: NodeStructure(activationFunction: Sigmoid(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 256))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: Sigmoid(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 128))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 64))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.1), count: 10))
var outputlayer = OutputLayer(outputSize: 10, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
var networkConfig = NNConfig(inputSize: 784, structure: networkStructure, outputLayer: outputlayer)
var trainingConfig = TrainingConfig(batchSize: 20, learningRate: 0.04, negativeAttempts: 5)
let network = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
network.setLabelsMeaning(use: labelsMeaning)

network.fp(input: trainingImage0, labels: trainingLabel0)
network.printResultsInDetail()

let startTime = Date()
for i in 0..<100 {
    print("processing batch: \(i + 1)/100")
    network.descentbatches(inputs: trainingImages.0, labels: trainingImages.1)
    network.fp(input: trainingImage0, labels: trainingLabel0)
    network.printResultsInDetail()
    network.fp(input: trainingImage1, labels: trainingLabel1)
    network.printResultsInDetail()
    network.fp(input: trainingImage2, labels: trainingLabel2)
    network.printResultsInDetail()
    network.fp(input: trainingImage3, labels: trainingLabel3)
    network.printResultsInDetail()
}
let endTime = Date()
print("Done! in \(endTime.timeIntervalSince(startTime))")

network.fp(input: trainingImage0, labels: trainingLabel0)
network.printResultsInDetail()

// network.fp(input: testImages.0[0], labels: testImages.1[0])
// network.printResults()
// let startTime = Date()
// network.descentbatches(inputs: trainingImages.0, labels: trainingImages.1)
// network.fp(input: testImages.0[0], labels: testImages.1[0])
// network.printResults()
// let endTime = Date()
// print("Done! in \(endTime.timeIntervalSince(startTime))")

// let startTime = Date()
// for i in 0..<100 {
//     print("\rProcessing: \(i + 1)/100", terminator: "")
//     fflush(stdout) // 刷新标准输出
//     network.fp(input: testImage1Dim, labels: testLabel)
// }
// let endTime = Date()
// print("\nDone! in \(endTime.timeIntervalSince(startTime))")

