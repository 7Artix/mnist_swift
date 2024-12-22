import Foundation

class CNN: NN {
    let moduleCNN: CNNModule
    init(networkConfig: NNConfig, trainingConfig: TrainingConfig, moduleCNN: CNNModule) {
        self.moduleCNN = moduleCNN
        let networkConfigCNN = NNConfig(inputSize: moduleCNN.outputSizeCNN, structure: networkConfig.structure, outputLayer: networkConfig.outputLayer)
        super.init(networkConfig: networkConfigCNN, trainingConfig: trainingConfig)
    }
}