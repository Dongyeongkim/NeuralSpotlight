import NeuralSpotlightKit
import Foundation

/// One vector-space identity for text, images and audio, all executed through Core ML.
/// Query workers prepare the common query graph before reporting ready. Index workers load media graphs lazily.
public final class CoreMLEmbedder: Embedder {
    public static let modelInfo = WorkerInfo(
        model: "google/embeddinggemma-2", revision: CoreMLModelFiles.revision,
        runtime: "coreml-all-text128-windows512-vision280-audio10-v2",
        dtype: "fp16-safe-rmsnorm+bf16-table", dim: 768, textOnly: false, loadSeconds: nil)

    public let info: WorkerInfo
    private let text: CoreMLTextEmbedder
    private var media: CoreMLMediaEmbedder?
    private var closed = false

    public init(folder: URL, textOnly: Bool) throws {
        let start = Date()
        guard CoreMLModelFiles.complete(in: folder, textOnly: textOnly) else {
            throw CoreMLError.invalidAsset("incomplete Core ML model in \(folder.path)")
        }
        // Verify the replacement assets before the indexing service retires a previous runtime's vectors.
        if !textOnly {
            try CoreMLModelCache.verify(folder: folder, files: CoreMLModelFiles.vision)
            try CoreMLModelCache.verify(folder: folder, files: CoreMLModelFiles.audio)
        }
        text = try CoreMLTextEmbedder(folder: folder, textOnly: textOnly)
        if textOnly {
            try text.prepareForQueries()
        } else {
            media = CoreMLMediaEmbedder(folder: folder, text: text)
        }
        let known = Self.modelInfo
        info = WorkerInfo(model: known.model, revision: known.revision, runtime: known.runtime, dtype: known.dtype,
                          dim: known.dim, textOnly: textOnly, loadSeconds: Date().timeIntervalSince(start))
    }

    public func embedTexts(_ texts: [String]) throws -> [[Float]] {
        guard !closed else { throw CoreMLError.unavailable("embedder is closed") }
        do { return try text.embedTexts(texts) }
        catch { throw WorkerError.failed("Core ML text embedding failed: \(error)") }
    }

    public func embedImages(_ paths: [String], budget: Int) throws -> [Embedding] {
        guard !closed, let media else { throw CoreMLError.unavailable("media is unavailable in a query worker") }
        return try media.embedImages(paths, budget: budget)
    }

    public func embedAudio(_ paths: [String]) throws -> [Embedding] {
        guard !closed, let media else { throw CoreMLError.unavailable("media is unavailable in a query worker") }
        return try media.embedAudio(paths)
    }

    public func close() {
        guard !closed else { return }
        media?.close()
        media = nil
        text.close()
        closed = true
    }

    deinit { close() }
}
