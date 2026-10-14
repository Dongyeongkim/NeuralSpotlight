import CoreML
import DigUpKit
import Foundation

/// EmbeddingGemma 2's shared text transformer. Core ML may use ANE, GPU and CPU; operator placement is automatic.
/// The media encoder passes its image/audio tokens through this same transformer to share the embedding space.
/// One caller at a time, like `Embedder`. Models load lazily per sequence length, and `close` releases them.
public final class CoreMLTextEmbedder: Embedder {
    public static let lengths = [32, 48, 64, 128, 256, 512]
    /// One prepared shape covers ordinary queries instead of loading a new graph at 32/48/64-token boundaries.
    static let queryLengths = [128, 512]
    public static let modelInfo = WorkerInfo(
        model: "google/embeddinggemma-2", revision: CoreMLModelFiles.revision,
        runtime: "coreml-all-text128-windows512-v2", dtype: "fp16-safe-rmsnorm+bf16-table",
        dim: 768, textOnly: true, loadSeconds: nil)
    public private(set) var info: WorkerInfo
    private var tokenizer: GemmaTokenizer?
    private var table: Data?
    private var models: [Int: MLModel] = [:]
    private var modelOrder: [Int] = []
    private let compiled: URL
    private let queryOnly: Bool

    public convenience init(textOnly: Bool = true) throws {
        try self.init(folder: CoreMLModelFiles.locate(textOnly: true), textOnly: textOnly)
    }

    public init(folder: URL, textOnly: Bool = true) throws {
        guard #available(macOS 15, *), CoreMLModelFiles.supported else {
            throw CoreMLError.unavailable("requires Apple silicon and macOS 15 or newer")
        }
        let start = Date()
        try CoreMLModelCache.verify(folder: folder, files: CoreMLModelFiles.text)
        coreMLLog("text verification", since: start)
        let table = try Data(contentsOf: folder.appendingPathComponent("embeddings.bf16"), options: .alwaysMapped)
        guard table.count == 262_144 * 512 * 2 else { throw CoreMLError.invalidAsset("invalid token table dimensions") }
        self.table = table
        let tokenStart = Date()
        tokenizer = try GemmaTokenizer(tokenizerJsonURL: folder.appendingPathComponent("tokenizer.json"))
        coreMLLog("tokenizer", since: tokenStart)
        let compileStart = Date()
        compiled = try CoreMLModelCache.compile(folder: folder, package: CoreMLModelFiles.package)
        coreMLLog("text compiled asset", since: compileStart)
        queryOnly = textOnly
        let known = Self.modelInfo
        info = WorkerInfo(model: known.model, revision: known.revision, runtime: known.runtime, dtype: known.dtype,
                          dim: known.dim, textOnly: textOnly, loadSeconds: Date().timeIntervalSince(start))
    }

    public func close() {
        models.removeAll()
        modelOrder.removeAll()
        table = nil
        tokenizer = nil
    }

    public func embedTexts(_ texts: [String]) throws -> [[Float]] {
        guard let tokenizer, table != nil else { throw CoreMLError.unavailable("embedder is closed") }
        return try texts.map { text in
            let tokens = tokenizer.tokenize(text)
            if queryOnly || tokens.count <= 510 {
                return try predict([tokenizer.bosId] + Array(tokens.prefix(510)) + [tokenizer.eosId])
            }
            // Existing extractors already produce document/code chunks. If one still exceeds the graph's fixed
            // limit, keep every body token using independent windows. Repeat the known document prefix in each.
            let prefix = Self.documentPrefix(text)
            let prefixTokens = tokenizer.tokenize(prefix)
            let shared = zip(prefixTokens, tokens).prefix(while: { $0 == $1 }).count
            let prefixCount = min(shared, 128)
            let chunks = Self.windows(tokens: tokens, prefixCount: prefixCount)
            let values = try chunks.map { chunk in
                try predict([tokenizer.bosId] + chunk.tokens + [tokenizer.eosId])
            }
            return try Self.combine(values, weights: chunks.map(\.weight))
        }
    }

    public func embedImages(_ paths: [String], budget: Int) throws -> [Embedding] {
        throw CoreMLError.unavailable("use the multimodal Core ML embedder for images")
    }

    public func embedAudio(_ paths: [String]) throws -> [Embedding] {
        throw CoreMLError.unavailable("use the multimodal Core ML embedder for audio")
    }

    /// Run once before a query worker reports ready. This includes device specialization and first prediction.
    public func prepareForQueries() throws {
        guard let tokenizer else { throw CoreMLError.unavailable("embedder is closed") }
        let start = Date()
        _ = try predict([tokenizer.bosId, tokenizer.eosId], forcedLength: 128)
        coreMLLog("query warm-up (128 tokens)", since: start)
    }

    public enum MediaModality { case image, audio }

    /// Media tokens are already projected/scaled by the encoder; only vocabulary tokens use the BF16 table scale.
    public func embedMediaTokens(_ tokens: MLMultiArray, modality: MediaModality, validCount: Int) throws -> [Float] {
        guard let tokenizer, tokens.shape.count == 3, tokens.shape[0].intValue == 1,
              tokens.shape[2].intValue == 512, validCount > 0, validCount <= tokens.shape[1].intValue,
              let length = Self.lengths.first(where: { $0 >= validCount + 4 }) else {
            throw CoreMLError.prediction("invalid media token shape")
        }
        let startToken = tokenizer.tokenize(modality == .image ? "<|image>" : "<|audio>")
        let endToken = tokenizer.tokenize(modality == .image ? "<image|>" : "<audio|>")
        guard startToken.count == 1, endToken.count == 1 else {
            throw CoreMLError.invalidAsset("missing media boundary tokens")
        }
        var ids = [tokenizer.bosId, startToken[0]]
        ids += [Int32](repeating: tokenizer.padId, count: validCount)
        ids += [endToken[0], tokenizer.eosId]
        let (embeds, mask) = try inputs(ids, length: length)
        let target = embeds.dataPointer.assumingMemoryBound(to: Float16.self)
        let read: (Int) -> Float
        switch tokens.dataType {
        case .float16:
            let values = tokens.dataPointer.assumingMemoryBound(to: Float16.self)
            read = { Float(values[$0]) }
        case .float32:
            let values = tokens.dataPointer.assumingMemoryBound(to: Float.self)
            read = { values[$0] }
        default:
            throw CoreMLError.prediction("unsupported media token precision")
        }
        for row in 0..<validCount {
            for column in 0..<512 {
                let offset = row * tokens.strides[1].intValue + column * tokens.strides[2].intValue
                let value = read(offset)
                let half = Float16(value)
                guard half.isFinite else { throw CoreMLError.prediction("media token exceeds FP16 range") }
                target[(row + 2) * 512 + column] = half
            }
        }
        return try predict(embeds: embeds, mask: mask, length: length)
    }

    struct Window { let tokens: [Int32]; let weight: Int }

    static func windows(tokens: [Int32], prefixCount: Int) -> [Window] {
        let prefixCount = min(max(prefixCount, 0), min(tokens.count, 128))
        let prefix = Array(tokens.prefix(prefixCount))
        let capacity = 510 - prefixCount
        let body = tokens.dropFirst(prefixCount)
        guard !body.isEmpty else { return [Window(tokens: prefix, weight: 1)] }
        return stride(from: body.startIndex, to: body.endIndex, by: capacity).map { start in
            let end = min(start + capacity, body.endIndex)
            return Window(tokens: prefix + body[start..<end], weight: end - start)
        }
    }

    static func documentPrefix(_ text: String) -> String {
        guard text.hasPrefix("title: "), let range = text.range(of: " | text:") else { return "" }
        return String(text[..<range.upperBound])
    }

    static func combine(_ vectors: [[Float]], weights: [Int]) throws -> [Float] {
        guard let first = vectors.first, vectors.count == weights.count, !first.isEmpty,
              vectors.allSatisfy({ $0.count == first.count }), weights.allSatisfy({ $0 > 0 }) else {
            throw CoreMLError.prediction("invalid window embeddings")
        }
        var result = [Float](repeating: 0, count: first.count)
        let total = Float(weights.reduce(0, +))
        for (vector, weight) in zip(vectors, weights) {
            for column in result.indices { result[column] += vector[column] * (Float(weight) / total) }
        }
        return try normalized(result)
    }

    static func normalized(_ vector: [Float]) throws -> [Float] {
        guard vector.allSatisfy(\.isFinite) else { throw CoreMLError.prediction("non-finite embedding") }
        let norm = vector.reduce(Double(0)) { $0 + Double($1) * Double($1) }.squareRoot()
        guard norm.isFinite, norm > 0 else { throw CoreMLError.prediction("zero embedding") }
        return vector.map { Float(Double($0) / norm) }
    }

    @available(macOS 15, *)
    private func function(length: Int) throws -> MLModel {
        if let model = models[length] {
            modelOrder.removeAll { $0 == length }
            modelOrder.append(length)
            return model
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.functionName = "embed_\(length)"
        let started = Date()
        let model: MLModel
        do { model = try MLModel(contentsOf: compiled, configuration: configuration) }
        catch { throw WorkerError.failed("Core ML text model could not load: \(error)") }
        coreMLLog("load embed_\(length) (all compute units)", since: started)
        models[length] = model
        modelOrder.append(length)
        // Indexing interleaves six input lengths across pages, code stretches and media. An OS-cached load
        // still costs about 0.5 s here, so evicting after two shapes spends far more time loading than predicting.
        // Retain the bounded six shapes for the indexing worker's lifetime; query workers only need two.
        // close() and the helper's idle exit release them together. Input shapes and vectors remain unchanged.
        let capacity = queryOnly ? Self.queryLengths.count : Self.lengths.count
        if modelOrder.count > capacity { models.removeValue(forKey: modelOrder.removeFirst()) }
        return model
    }

    private func predict(_ ids: [Int32], forcedLength: Int? = nil) throws -> [Float] {
        guard let length = forcedLength ?? (queryOnly ? Self.queryLengths : Self.lengths).first(where: { $0 >= ids.count }) else {
            throw CoreMLError.prediction("invalid token window")
        }
        let (embeds, mask) = try inputs(ids, length: length)
        return try predict(embeds: embeds, mask: mask, length: length)
    }

    private func inputs(_ ids: [Int32], length: Int) throws -> (MLMultiArray, MLMultiArray) {
        guard let table, let tokenizer, !ids.isEmpty, ids.count <= length else {
            throw CoreMLError.prediction("invalid token window")
        }
        let embeds = try MLMultiArray(shape: [1, NSNumber(value: length), 512], dataType: .float16)
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float16)
        let rows = table.count / (512 * 2)
        let target = embeds.dataPointer.assumingMemoryBound(to: Float16.self)
        let maskPointer = mask.dataPointer.assumingMemoryBound(to: Float16.self)
        try table.withUnsafeBytes { raw in
            let values = raw.bindMemory(to: UInt16.self)
            for position in 0..<length {
                let id = position < ids.count ? ids[position] : tokenizer.padId
                guard id >= 0, id < rows else { throw CoreMLError.invalidAsset("token id \(id) outside table") }
                for column in 0..<512 {
                    let bf16 = UInt16(littleEndian: values[Int(id) * 512 + column])
                    target[position * 512 + column] = Float16(Float(bitPattern: UInt32(bf16) << 16) * Float(512).squareRoot())
                }
                maskPointer[position] = position < ids.count ? 1 : 0
            }
        }
        return (embeds, mask)
    }

    private func predict(embeds: MLMultiArray, mask: MLMultiArray, length: Int) throws -> [Float] {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "inputs_embeds": MLFeatureValue(multiArray: embeds), "attention_mask": MLFeatureValue(multiArray: mask),
        ])
        let output = try function(length: length).prediction(from: input)
        guard let vector = output.featureValue(for: "embedding")?.multiArrayValue, vector.count == 768 else {
            throw CoreMLError.prediction("missing 768-dimensional output")
        }
        return try Self.normalized((0..<768).map { vector[$0].floatValue })
    }

}

func coreMLLog(_ phase: String, since start: Date) {
    let line = "Core ML: \(phase) \(Int(Date().timeIntervalSince(start) * 1000)) ms\n"
    FileHandle.standardError.write(Data(line.utf8))
}
