import Foundation
import Testing
@testable import NeuralSpotlightKit

struct EmbeddingBackendTests {
    @Test func modelURLsKeepRemoteAndInstalledPathsSeparate() {
        let file = ModelFiles.File(name: "CoreML/model.mlpackage/Manifest.json", bytes: 100,
                                   sha256: "hash", repository: "owner/model", revision: "fixed",
                                   remotePath: "model.mlpackage/Manifest.json")
        #expect(file.url.absoluteString == "https://huggingface.co/owner/model/resolve/fixed/model.mlpackage/Manifest.json")
        #expect(CoreMLModelFiles.all.allSatisfy { $0.repository == CoreMLModelFiles.repository })
    }

    @Test func automaticNeverFallsBackWhenCoreMLAssetsAreMissing() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(EmbeddingBackend.automatic.resolved(in: root) == .coreML)
        #expect(EmbeddingBackend.coreML.resolved(in: root) == .coreML) // explicit mode must report missing files
        #expect(!EmbeddingBackend.coreMLReady(in: root))
        #expect(EmbeddingBackend(rawValue: "llama") == nil)
        #if os(macOS)
        guard EmbeddingBackend.supportsCoreML else { return }
        for file in CoreMLModelFiles.all {
            let path = root.appendingPathComponent(file.name)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: path.path, contents: nil)
            let handle = try FileHandle(forWritingTo: path)
            try handle.truncate(atOffset: UInt64(file.bytes)) // sparse fixtures; no weights loaded
            try handle.close()
        }
        #expect(EmbeddingBackend.automatic.resolved(in: root) == .coreML)
        #expect(EmbeddingBackend.coreMLReady(in: root))
        try Data().write(to: root.appendingPathComponent(CoreMLModelFiles.all[0].name))
        #expect(!EmbeddingBackend.coreMLReady(in: root))
        #expect(EmbeddingBackend.automatic.resolved(in: root) == .coreML)
        #endif
    }

    @Test func unsupportedPlatformsDoNotAdvertiseNativeDownloads() {
        guard !EmbeddingBackend.supportsCoreML else { return }
        #expect(EmbeddingBackend.automatic.modelFiles.isEmpty)
        #expect(EmbeddingBackend.coreML.modelFiles.isEmpty)
    }
}
