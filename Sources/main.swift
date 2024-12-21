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

// do {
//     let imageRecipeGray8 = ImageRecipe(width: 28, height: 28, pixelFormat: .GRAY8, provider: try .makeProviderGrayscale(from: firstImage))
//     let cgImageGray8 = try CGImage.createCGImage(imageRecipe: imageRecipeGray8)
//     try cgImageGray8.saveCGImage(useName: "first_image_label_\(firstLabel)", toPath: "./test", as: .png)
// }

let batchSize = 40

//let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: batchSize)
let testImages = mnistTest.getImagesBatchForNetwork(fromIndex: 0, batchSize: 50)

// do {
//     for i in 0..<20 {
//         let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: i * batchSize, batchSize: batchSize)
//         let image1Dim = trainingImages.images[0]
//         var image2Dim: [[UInt8]] = []
//         for i in 0..<28 {
//             let row = Array(image1Dim[i * 28..<(i + 1) * 28]).map { UInt8($0) }
//             image2Dim.append(row)
//         }
//         let label = trainingImages.labels[0]
//         let labelIndex: Int = Int(label.firstIndex(of: Double(1.0))!)
//         let imageRecipeGray8 = ImageRecipe(width: 28, height: 28, pixelFormat: .GRAY8, provider: try .makeProviderGrayscale(from: image2Dim))
//         let cgImageGray8 = try CGImage.createCGImage(imageRecipe: imageRecipeGray8)
//         try cgImageGray8.saveCGImage(useName: "image_\(i)_label_\(labelIndex)", toPath: "./test", as: .png)
//     }
// }

// let trainingImage0 = trainingImages.images[0]
// let trainingLabel0 = trainingImages.labels[0]
// let trainingImage1 = trainingImages.images[1]
// let trainingLabel1 = trainingImages.labels[1]
// let trainingImage2 = trainingImages.images[2]
// let trainingLabel2 = trainingImages.labels[2]
// let trainingImage3 = trainingImages.images[3]
// let trainingLabel3 = trainingImages.labels[3]

let labelsMeaning = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"]

var networkStructure: [[NodeStructure]] = []
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 512))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 256))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 128))
networkStructure.append(Array(repeating: NodeStructure(activationFunction: ReLU(), weightInitializer: heInitializer(inputSize:_:), bias: 0.01), count: 10))
var outputlayer = OutputLayer(outputSize: 10, normalizationFunction: Softmax(), lossFunction: CrossEntropy())
var networkConfig = NNConfig(inputSize: 784, structure: networkStructure, outputLayer: outputlayer)
var trainingConfig = TrainingConfig(batchSize: batchSize, epochSize: 500, learningRateBase: 0.1, learningRateScheduler: ExponentialDecay(), negativeAttempts: 5)
let network = NN(networkConfig: networkConfig, trainingConfig: trainingConfig)
network.setLabelsMeaning(use: labelsMeaning)
network.setGradientThreshold(threshold: 2.0)

var trainingImages: [(images: [[Double]], labels: [[Double]])] = []
for i in 0..<500 {
    trainingImages.append(mnistTraining.getImagesBatchForNetwork(fromIndex: i * batchSize, batchSize: batchSize))
}

network.descentEpoch(imagesTraining: trainingImages, imagesTest: testImages)

for i in 0..<500 {
    print("processing batch: \(i + 1)/500")
    //let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: 0, batchSize: batchSize)
    //print(trainingImages.labels[i])
    let trainingImages = mnistTraining.getImagesBatchForNetwork(fromIndex: i * batchSize, batchSize: batchSize)
    network.descentbatches(inputs: trainingImages.images, labels: trainingImages.labels)
    //network.fp(input: trainingImages.images[0], labels: trainingImages.labels[0])
    //network.descentSingleStep()
    network.printAccuracy(ImgaesTest: testImages.images, LabelsTest: testImages.labels)
}

// network.fp(input: trainingImage0, labels: trainingLabel0)
// network.printResultsInDetail()

