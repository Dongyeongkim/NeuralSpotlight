import Foundation

/// Core ML is the native runtime. Automatic is retained as a settings/CLI alias; missing files never change runtimes.
public enum EmbeddingBackend: String, CaseIterable, Sendable {
    case automatic = "auto"
    case coreML = "coreml"

    public static var environment: Self {
        Self(rawValue: ProcessInfo.processInfo.environment["DIGUP_EMBEDDING_BACKEND"] ?? "auto") ?? .automatic
    }

    public static var supportsCoreML: Bool {
        #if os(macOS) && arch(arm64)
        if #available(macOS 15, *) { return true }
        #endif
        return false
    }

    /// Files for the native app's model download, all relative to its Models folder.
    public var modelFiles: [ModelFiles.File] {
        Self.supportsCoreML ? CoreMLModelFiles.all : []
    }

    public static func coreMLReady(in modelsFolder: URL) -> Bool {
        #if os(macOS)
        guard supportsCoreML else { return false }
        return CoreMLModelFiles.all.allSatisfy {
            FileFetcher.size(of: modelsFolder.appendingPathComponent($0.name)) == $0.bytes
        }
        #else
        return false
        #endif
    }

    public func resolved(in modelsFolder: URL) -> Self {
        .coreML
    }
}
