import DigUpKit
import Foundation
#if os(macOS)
import CoreMLRuntime
#endif

/// One native runtime and vector identity for the helper's entire lifetime.
enum EmbeddingRuntime {
    static var preference: EmbeddingBackend {
        let arguments = ProcessInfo.processInfo.arguments
        if let option = arguments.firstIndex(of: "--backend"), option + 1 < arguments.count,
           let value = EmbeddingBackend(rawValue: arguments[option + 1]) { return value }
        return .environment
    }

    static let backend = EmbeddingBackend.coreML
    static let unavailable = "Native embedding requires Apple silicon and macOS 15 or newer; configure DIGUP_WORKER_DIR for an external embedding worker on this platform"

    static var modelInfo: WorkerInfo? {
        #if os(macOS)
        return CoreMLEmbedder.modelInfo
        #else
        return nil
        #endif
    }

    static var ready: Bool { ready(textOnly: false) }

    static func ready(textOnly: Bool) -> Bool {
        if ProcessInfo.processInfo.environment["DIGUP_WORKER_DIR"] != nil { return true }
        guard EmbeddingBackend.supportsCoreML else { return false }
        return textOnly ? CoreMLModelFiles.textComplete : CoreMLModelFiles.complete
    }

    static func requireSupport() throws {
        guard ProcessInfo.processInfo.environment["DIGUP_WORKER_DIR"] != nil || EmbeddingBackend.supportsCoreML else {
            throw WorkerError.notFound(unavailable)
        }
    }

    static func make(textOnly: Bool, log: URL?) throws -> any Embedder {
        #if os(macOS)
        guard EmbeddingBackend.supportsCoreML else { throw WorkerError.notFound(unavailable) }
        return try CoreMLEmbedder(folder: CoreMLModelFiles.folder, textOnly: textOnly)
        #else
        throw WorkerError.notFound(unavailable)
        #endif
    }
}
