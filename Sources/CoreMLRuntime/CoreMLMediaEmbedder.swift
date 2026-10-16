// Audio preprocessing adapted from FluidInference/FluidUse (Apache-2.0), revision
// d278e17df85e39e8c21bb64a467da81c39576fcd, EmbeddingGemma2Audio.swift.
// Changes: synchronous streaming, bounded memory, all compute units, per-file errors and complete audio tails.
// License: FluidUse-LICENSE.txt in this directory.
@preconcurrency import AVFoundation
import CoreGraphics
import CoreML
import NeuralSpotlightKit
import Foundation
import ImageIO

/// Core ML vision/audio towers feed their projected tokens into the same text encoder as queries.
/// Core ML may use CPU, GPU or Neural Engine; models load lazily, with one vision budget resident at a time.
/// A caller serializes access, as it does for `Embedder`.
public final class CoreMLMediaEmbedder {
    public static let budgets = [70, 140, 280]
    static let sampleRate = 16_000
    static let windowSamples = 160_000
    static let leftPad = 160
    static let frames = 1_000

    private let folder: URL
    private let text: CoreMLTextEmbedder
    private var vision: (budget: Int, model: MLModel)?
    private var audio: MLModel?
    private var positions: Data?
    private var visionCompiled: URL?
    private var closed = false

    public init(folder: URL, text: CoreMLTextEmbedder) {
        self.folder = folder
        self.text = text
    }

    public func close() {
        vision = nil
        audio = nil
        positions = nil
        visionCompiled = nil
        closed = true
    }

    public func embedImages(_ paths: [String], budget: Int) throws -> [Embedding] {
        guard !closed else { throw CoreMLError.unavailable("media embedder is closed") }
        guard Self.budgets.contains(budget) else {
            throw CoreMLError.prediction("Core ML image budget must be one of \(Self.budgets)")
        }
        guard !paths.isEmpty else { return [] }
        let model: MLModel
        do { model = try visionModel(budget: budget) }
        catch { throw WorkerError.failed("Core ML vision model could not load: \(error)") }
        guard let positions else { throw WorkerError.failed("Core ML vision position table is unavailable") }
        return try paths.map { path in
            try embedding {
                let image = try Self.readImage(URL(fileURLWithPath: path))
                let input = try Self.visionInput(image: image, budget: budget, positions: positions)
                let output = try model.prediction(from: input.features)
                guard let tokens = output.featureValue(for: "image_tokens")?.multiArrayValue else {
                    throw CoreMLError.prediction("missing image_tokens output")
                }
                return try text.embedMediaTokens(tokens, modality: .image, validCount: input.tokenCount)
            }
        }
    }

    public func embedAudio(_ paths: [String]) throws -> [Embedding] {
        guard !closed else { throw CoreMLError.unavailable("media embedder is closed") }
        guard !paths.isEmpty else { return [] }
        let model: MLModel
        do { model = try audioModel() }
        catch { throw WorkerError.failed("Core ML audio model could not load: \(error)") }
        return try paths.map { path in try embedding { try embedRecording(URL(fileURLWithPath: path), model: model) } }
    }

    private func embedding(_ make: () throws -> [Float]) throws -> Embedding {
        do { return Embedding(vector: try make(), error: nil) }
        catch let error as WorkerError { throw error }
        catch { return Embedding(vector: nil, error: String("\(error)".prefix(300))) }
    }

    private func visionModel(budget: Int) throws -> MLModel {
        guard #available(macOS 15, *) else { throw CoreMLError.unavailable("vision requires macOS 15 or newer") }
        if let vision, vision.budget == budget { return vision.model }
        let started = Date()
        if visionCompiled == nil {
            try CoreMLModelCache.verify(folder: folder, files: CoreMLModelFiles.vision)
            let data = try Data(contentsOf: folder.appendingPathComponent("position_embeddings.f16"), options: .alwaysMapped)
            guard data.count == 2 * 1024 * 768 * 2 else {
                throw CoreMLError.invalidAsset("invalid vision position table dimensions")
            }
            positions = data
            visionCompiled = try CoreMLModelCache.compile(folder: folder, package: CoreMLModelFiles.visionPackage)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.functionName = "vision_\(budget)"
        vision = nil // Release the previous budget before loading another large vision graph.
        let model = try MLModel(contentsOf: visionCompiled!, configuration: configuration)
        vision = (budget, model)
        coreMLLog("vision_\(budget) load (.all)", since: started)
        return model
    }

    private func audioModel() throws -> MLModel {
        if let audio { return audio }
        let started = Date()
        try CoreMLModelCache.verify(folder: folder, files: CoreMLModelFiles.audio)
        let compiled = try CoreMLModelCache.compile(folder: folder, package: CoreMLModelFiles.audioPackage)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        audio = model
        coreMLLog("audio load (.all)", since: started)
        return model
    }

    // MARK: Images

    static func readImage(_ url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL,
                [kCGImageSourceShouldCache: false] as CFDictionary),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width > 0, image.height > 0 else {
            throw CoreMLError.invalidAsset("cannot decode image: \(url.lastPathComponent)")
        }
        return image
    }

    static func rgb(_ image: CGImage) throws -> [UInt8] {
        guard image.width <= 32768, image.height <= 32768, image.width * image.height <= 100_000_000 else {
            throw CoreMLError.invalidAsset("image is too large to decode; prepare a smaller copy")
        }
        var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
        let made = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
            return true
        }
        guard made else { throw CoreMLError.invalidAsset("cannot convert image to RGB") }
        var rgb = [UInt8](repeating: 0, count: image.width * image.height * 3)
        for pixel in 0..<(image.width * image.height) {
            for channel in 0..<3 { rgb[pixel * 3 + channel] = rgba[pixel * 4 + channel] }
        }
        return rgb
    }

    struct VisionInput {
        let features: MLDictionaryFeatureProvider
        let tokenCount: Int
        let height: Int
        let width: Int
    }

    /// Row-major 16×16 RGB patches, their x/y positions, and 3×3 patch pooling, exactly as the exported graph expects.
    static func visionInput(image: CGImage, budget: Int, positions table: Data) throws -> VisionInput {
        guard Self.budgets.contains(budget), table.count == 2 * 1024 * 768 * 2,
              let size = ImageSizing.targetSize(height: image.height, width: image.width, budget: budget) else {
            throw CoreMLError.invalidAsset("invalid vision input or token budget")
        }
        let pixels = try rgb(image).withUnsafeBufferPointer {
            ImageSizing.resizeBicubic($0.baseAddress!, width: image.width, height: image.height,
                                     toWidth: size.width, toHeight: size.height)
        }
        let columns = size.width / 16, rows = size.height / 16, capacity = budget * 9
        let count = rows * columns, tokenCount = (rows / 3) * (columns / 3)
        guard count > 0, count <= capacity, columns <= 1024, rows <= 1024, tokenCount <= budget else {
            throw CoreMLError.invalidAsset("vision patch grid exceeds the exported model")
        }
        let patches = try MLMultiArray(shape: [1, NSNumber(value: capacity), 768], dataType: .float16)
        let coordinates = try MLMultiArray(shape: [1, NSNumber(value: capacity), 2], dataType: .int32)
        let embeddings = try MLMultiArray(shape: [1, NSNumber(value: capacity), 768], dataType: .float16)
        let valid = try MLMultiArray(shape: [1, NSNumber(value: capacity)], dataType: .float16)
        let pooling = try MLMultiArray(shape: [NSNumber(value: budget), NSNumber(value: capacity)], dataType: .float16)
        let patch = patches.dataPointer.assumingMemoryBound(to: Float16.self)
        let xy = coordinates.dataPointer.assumingMemoryBound(to: Int32.self)
        let position = embeddings.dataPointer.assumingMemoryBound(to: Float16.self)
        let mask = valid.dataPointer.assumingMemoryBound(to: Float16.self)
        let pool = pooling.dataPointer.assumingMemoryBound(to: Float16.self)
        patch.update(repeating: 0, count: capacity * 768)
        xy.update(repeating: -1, count: capacity * 2)
        position.update(repeating: 0, count: capacity * 768)
        mask.update(repeating: 0, count: capacity)
        pool.update(repeating: 0, count: budget * capacity)
        table.withUnsafeBytes { raw in
            let values = raw.bindMemory(to: UInt16.self)
            for y in 0..<rows {
                for x in 0..<columns {
                    let index = y * columns + x
                    xy[index * 2] = Int32(x)
                    xy[index * 2 + 1] = Int32(y)
                    mask[index] = 1
                    for column in 0..<768 {
                        let px = column / 3 % 16, py = column / (16 * 3), channel = column % 3
                        patch[index * 768 + column] = Float16(Float(pixels[((y * 16 + py) * size.width + x * 16 + px) * 3 + channel]) / 255)
                        let horizontal = Float16(bitPattern: UInt16(littleEndian: values[x * 768 + column]))
                        let vertical = Float16(bitPattern: UInt16(littleEndian: values[(1024 + y) * 768 + column]))
                        position[index * 768 + column] = horizontal + vertical
                    }
                    let group = (y / 3) * (columns / 3) + x / 3
                    pool[group * capacity + index] = Float16(1.0 / 9.0)
                }
            }
        }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "patches": MLFeatureValue(multiArray: patches), "positions": MLFeatureValue(multiArray: coordinates),
            "position_embeddings": MLFeatureValue(multiArray: embeddings), "valid": MLFeatureValue(multiArray: valid),
            "pool": MLFeatureValue(multiArray: pooling),
        ])
        return VisionInput(features: features, tokenCount: tokenCount, height: size.height, width: size.width)
    }

    // MARK: Audio

    static func validFrames(samples: Int) -> Int {
        let real = leftPad + max(0, min(samples, windowSamples))
        return real < 321 ? 0 : min(frames, (real - 321) / 160 + 1)
    }

    static func tokenCount(validFrames: Int) -> Int { (max(0, validFrames) + 3) / 4 }

    static func audioInput(_ samples: [Float]) throws -> (features: MLDictionaryFeatureProvider, tokens: Int) {
        guard !samples.isEmpty, samples.count <= windowSamples, samples.allSatisfy(\.isFinite) else {
            throw CoreMLError.invalidAsset("invalid audio window")
        }
        let waveform = try MLMultiArray(shape: [1, NSNumber(value: windowSamples + leftPad)], dataType: .float32)
        let frameMask = try MLMultiArray(shape: [1, NSNumber(value: frames)], dataType: .float16)
        let wave = waveform.dataPointer.assumingMemoryBound(to: Float.self)
        wave.update(repeating: 0, count: windowSamples + leftPad)
        samples.withUnsafeBufferPointer { (wave + leftPad).update(from: $0.baseAddress!, count: samples.count) }
        // Preserve even a very short final tail, padding it just enough for one valid frame.
        let valid = validFrames(samples: max(161, samples.count))
        let mask = frameMask.dataPointer.assumingMemoryBound(to: Float16.self)
        for frame in 0..<frames { mask[frame] = frame < valid ? 1 : 0 }
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: waveform), "frame_mask": MLFeatureValue(multiArray: frameMask),
        ])
        return (features, tokenCount(validFrames: valid))
    }

    private func embedRecording(_ url: URL, model: MLModel) throws -> [Float] {
        var pending: [Float] = [], combined = [Double](repeating: 0, count: 768)
        var total = 0
        func consume(_ samples: [Float]) throws {
            let input = try Self.audioInput(samples)
            let output = try model.prediction(from: input.features)
            guard let tokens = output.featureValue(for: "audio_tokens")?.multiArrayValue else {
                throw CoreMLError.prediction("missing audio_tokens output")
            }
            let vector = try text.embedMediaTokens(tokens, modality: .audio, validCount: input.tokens)
            for column in combined.indices { combined[column] += Double(vector[column]) * Double(samples.count) }
            total += samples.count
        }
        try Self.readAudio(url) { samples in
            pending.append(contentsOf: samples)
            while pending.count >= Self.windowSamples {
                try consume(Array(pending.prefix(Self.windowSamples)))
                pending.removeFirst(Self.windowSamples)
            }
        }
        if !pending.isEmpty { try consume(pending) }
        guard total > 0 else { throw CoreMLError.invalidAsset("audio file contains no samples") }
        return try CoreMLTextEmbedder.normalized(combined.map { Float($0 / Double(total)) })
    }

    /// Decode in bounded chunks. Indexer normally supplies 30-second WAV pieces; longer direct inputs are streamed.
    static func readAudio(_ url: URL, onSamples: ([Float]) throws -> Void) throws {
        let file = try AVAudioFile(forReading: url)
        // Preserve channels through resampling, then average them explicitly. AVAudioConverter's default
        // channel mapping can select only the first channel when asked for mono, losing the others entirely.
        guard file.processingFormat.channelCount > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
                                        channels: file.processingFormat.channelCount, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: target) else {
            throw CoreMLError.invalidAsset("cannot convert audio to 16 kHz mono")
        }
        let chunk: AVAudioFrameCount = 65_536
        let capacity = AVAudioFrameCount(Double(chunk) * Double(sampleRate) / file.processingFormat.sampleRate) + 1024
        while true {
            guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else {
                throw CoreMLError.invalidAsset("cannot allocate audio conversion buffer")
            }
            var conversionError: NSError?, readError: Error?
            let status = converter.convert(to: output, error: &conversionError) { _, outStatus in
                do {
                    // macOS 27 reports eofErr even when a read merely extends past the last frame. Bound the
                    // final read and signal end-of-stream while the converter drains its remaining output.
                    let remaining = file.length - file.framePosition
                    guard remaining > 0 else { outStatus.pointee = .endOfStream; return nil }
                    guard let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: chunk) else {
                        throw CoreMLError.invalidAsset("cannot allocate audio input buffer")
                    }
                    try file.read(into: input, frameCount: AVAudioFrameCount(min(remaining, AVAudioFramePosition(chunk))))
                    guard input.frameLength > 0 else { outStatus.pointee = .endOfStream; return nil }
                    outStatus.pointee = .haveData
                    return input
                } catch {
                    readError = error
                    outStatus.pointee = .endOfStream
                    return nil
                }
            }
            if let readError { throw readError }
            if let conversionError { throw conversionError }
            if status == .error { throw CoreMLError.invalidAsset("audio conversion failed") }
            if let channels = output.floatChannelData, output.frameLength > 0 {
                let count = Int(output.frameLength), channelCount = Int(output.format.channelCount)
                var mono = [Float](repeating: 0, count: count)
                let weight = Float(1) / Float(channelCount)
                for channel in 0..<channelCount {
                    for sample in 0..<count { mono[sample] += channels[channel][sample] * weight }
                }
                try onSamples(mono)
            }
            if status == .endOfStream { return }
        }
    }
}
