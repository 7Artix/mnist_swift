import Foundation

class Database {
    enum DatabasePath {
        case TrainingImages
        case TrainingLabels
        case TestImages
        case TestLabels
    }

    var trainingImagesPath = "./mnist_database/train-images.idx3-ubyte"
    var trainingLabelsPath = "./mnist_database/train-labels.idx1-ubyte"
    var testImagesPath = "./mnist_database/t10k-images.idx3-ubyte"
    var testLabelsPath = "./mnist_database/t10k-labels.idx1-ubyte"

    var trainingImagesData: Data
    var trainingLabelsData: Data
    var testImagesData: Data
    var testLabelsData: Data

    init(trainingImagesPath: String = "./mnist_database/train-images.idx3-ubyte", 
    trainingLabelsPath: String = "./mnist_database/train-labels.idx1-ubyte", 
    testImagesPath: String = "./mnist_database/t10k-images.idx3-ubyte", 
    testLabelsPath: String = "./mnist_database/t10k-labels.idx1-ubyte") {
        do {
            self.trainingImagesData = try Data(contentsOf: URL(fileURLWithPath: trainingImagesPath))
            self.trainingLabelsData = try Data(contentsOf: URL(fileURLWithPath: trainingLabelsPath))
            self.testImagesData = try Data(contentsOf: URL(fileURLWithPath: testImagesPath))
            self.testLabelsData = try Data(contentsOf: URL(fileURLWithPath: testLabelsPath))
        } catch {
            fatalError("Error: Load failed.")
        }
    }

    func setPath(for target: DatabasePath, with path: String) {
        switch target {
        case .TrainingImages:
            self.trainingImagesPath = path
        case .TrainingLabels:
            self.trainingLabelsPath = path
        case .TestImages:
            self.testImagesPath = path
        case .TestLabels:
            self.testLabelsPath = path
        }
    }

    func reloadDatabase() {
        do {
            self.trainingImagesData = try Data(contentsOf: URL(fileURLWithPath: trainingImagesPath))
            self.trainingLabelsData = try Data(contentsOf: URL(fileURLWithPath: trainingLabelsPath))
            self.testImagesData = try Data(contentsOf: URL(fileURLWithPath: testImagesPath))
            self.testLabelsData = try Data(contentsOf: URL(fileURLWithPath: testLabelsPath))
        } catch {
            fatalError("Error: Load failed.")
        }
    }
}
