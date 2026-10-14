import Foundation

public enum CoreMLError: Error, CustomStringConvertible {
    case invalidAsset(String), unavailable(String), prediction(String)

    public var description: String {
        switch self {
        case .invalidAsset(let why): "Core ML model: \(why)"
        case .unavailable(let why): "Core ML unavailable: \(why)"
        case .prediction(let why): "Core ML inference: \(why)"
        }
    }
}
