import Foundation

class NN {
    private enum ParameterType: String {
        case weight = "w"
        case bias = "b"
        case activation = "a"
    }
    private typealias Parameter = [ParameterType: [[Double]]]
    private var parameterTemp: Parameter = [:]
    private var parameterBest: Parameter = [:]
}
