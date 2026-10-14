import AVFoundation
import CoreGraphics
import CoreML
import DigUpKit
import Foundation
import Testing
@testable import CoreMLRuntime

@Suite struct CoreMLMediaPreprocessingTests {
    @Test func audioMasksMatchExportAndKeepShortTails() throws {
        #expect(CoreMLMediaEmbedder.validFrames(samples: 160_000) == 999)
        #expect(CoreMLMediaEmbedder.tokenCount(validFrames: 999) == 250)
        #expect(CoreMLMediaEmbedder.validFrames(samples: 64_000) == 399)
        #expect(CoreMLMediaEmbedder.tokenCount(validFrames: 399) == 100)
        #expect(CoreMLMediaEmbedder.validFrames(samples: 160) == 0)
        #expect(CoreMLMediaEmbedder.validFrames(samples: 161) == 1)
        let input = try CoreMLMediaEmbedder.audioInput([0.25, -0.5])
        #expect(input.tokens == 1)
        let wave = try #require(input.features.featureValue(for: "waveform")?.multiArrayValue)
        let mask = try #require(input.features.featureValue(for: "frame_mask")?.multiArrayValue)
        #expect(wave.count == 160_160)
        #expect((0..<160).allSatisfy { wave[$0].floatValue == 0 })
        #expect(wave[160].floatValue == 0.25 && wave[161].floatValue == -0.5)
        #expect(wave[162].floatValue == 0)
        #expect(mask[0].floatValue == 1 && mask[1].floatValue == 0)
        #expect(throws: CoreMLError.self) { try CoreMLMediaEmbedder.audioInput([]) }
        #expect(throws: CoreMLError.self) { try CoreMLMediaEmbedder.audioInput([.nan]) }
    }

    @Test func visionPatchesCoordinatesAndPoolingRetainPixelOrder() throws {
        // This aspect ratio exactly fills a 480×288 resize at budget 70: 540 patches, 60 real soft tokens.
        let width = 480, height = 288
        var pixels = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                pixels[(y * width + x) * 3] = UInt8(x % 256)
                pixels[(y * width + x) * 3 + 1] = UInt8(y % 256)
                pixels[(y * width + x) * 3 + 2] = 127
            }
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let image = try #require(CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: width * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: [],
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        var table = [Float16](repeating: 0, count: 2 * 1024 * 768)
        for row in 0..<1024 {
            table[row * 768] = Float16(row)
            table[(1024 + row) * 768] = Float16(row * 2)
        }
        let data = table.withUnsafeBytes { Data($0) }
        let input = try CoreMLMediaEmbedder.visionInput(image: image, budget: 70, positions: data)
        #expect(input.width == width && input.height == height && input.tokenCount == 60)
        let patches = try #require(input.features.featureValue(for: "patches")?.multiArrayValue)
        let positions = try #require(input.features.featureValue(for: "positions")?.multiArrayValue)
        let embeddings = try #require(input.features.featureValue(for: "position_embeddings")?.multiArrayValue)
        let valid = try #require(input.features.featureValue(for: "valid")?.multiArrayValue)
        let pool = try #require(input.features.featureValue(for: "pool")?.multiArrayValue)
        #expect(abs(patches[2].floatValue - 127.0 / 255.0) < 0.001)
        #expect(abs(patches[3].floatValue - 1.0 / 255.0) < 0.001)
        #expect(abs(patches[16 * 3 + 1].floatValue - 1.0 / 255.0) < 0.001)
        #expect(abs(patches[768].floatValue - 16.0 / 255.0) < 0.001)
        #expect(positions[2].intValue == 1 && positions[3].intValue == 0)
        #expect(positions[30 * 2].intValue == 0 && positions[30 * 2 + 1].intValue == 1)
        #expect(positions[540 * 2].intValue == -1 && positions[540 * 2 + 1].intValue == -1)
        #expect(embeddings[(30 + 1) * 768].floatValue == 3)
        #expect((0..<540).allSatisfy { valid[$0].floatValue == 1 })
        #expect((540..<630).allSatisfy { valid[$0].floatValue == 0 })
        let firstGroup = (0..<630).filter { pool[$0].floatValue != 0 }
        #expect(firstGroup == [0, 1, 2, 30, 31, 32, 60, 61, 62])
        #expect(abs(firstGroup.reduce(Float(0)) { $0 + pool[$1].floatValue } - 1) < 0.001)
        #expect((60 * 630..<70 * 630).allSatisfy { pool[$0].floatValue == 0 })
    }

    @Test func audioDecoderReadsAllSamplesAcrossMultipleModelWindows() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let count = 480_317 // More than three model windows and a nonempty tail.
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                               channels: 1, interleaved: false))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)))
        buffer.frameLength = AVAudioFrameCount(count)
        let pointer = try #require(buffer.floatChannelData?[0])
        for index in 0..<count { pointer[index] = Float(index % 100) / 200 }
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
            if #available(macOS 15, *) { file.close() }
        }
        var read: [Float] = []
        try CoreMLMediaEmbedder.readAudio(url) { read += $0 }
        #expect(read.count == count)
        #expect(read.last == Float((count - 1) % 100) / 200)
    }

    @Test func stereoAudioIsMixedToMonoAndDecodeErrorsAreReported() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let count = 16_317
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                               channels: 2, interleaved: false))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)))
        buffer.frameLength = AVAudioFrameCount(count)
        let channels = try #require(buffer.floatChannelData)
        channels[0].update(repeating: 0.25, count: count)
        channels[1].update(repeating: -0.25, count: count)
        do {
            let file = try AVAudioFile(forWriting: url, settings: format.settings)
            try file.write(from: buffer)
            if #available(macOS 15, *) { file.close() }
        }
        var read: [Float] = []
        try CoreMLMediaEmbedder.readAudio(url) { read += $0 }
        #expect(read.count == count)
        #expect(read.allSatisfy { abs($0) < 0.0001 })
        let missing = url.deletingLastPathComponent().appendingPathComponent(UUID().uuidString + ".wav")
        #expect(throws: (any Error).self) { try CoreMLMediaEmbedder.readAudio(missing) { _ in } }
    }
}

/// Existing generated image/speech fixtures and their previous runtime vectors catch preprocessing mistakes.
/// The new Core ML FP16 media graphs deliberately have a new vector identity, so exact byte equality is not expected.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["DIGUP_TEST_COREML"] == "1"), .serialized)
struct CoreMLMediaInferenceTests {
    static let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Reference/media")

    @Test func imagesAudioAndVideoFramesRemainSearchable() throws {
        let text = try CoreMLTextEmbedder(folder: GemmaTokenizerTests.modelFolder, textOnly: false)
        let media = CoreMLMediaEmbedder(folder: GemmaTokenizerTests.modelFolder, text: text)
        defer { media.close(); text.close() }
        struct Reference: Decodable { let vectors: [String: [Float]] }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: Self.folder.appendingPathComponent("vectors.json")))
        for (name, path, budget) in [("frame-70", "photo.jpg", 70), ("photo-140", "photo.jpg", 140),
                                      ("screen-280", "screen.png", 280)] {
            let results = try media.embedImages([Self.folder.appendingPathComponent(path).path,
                                                Self.folder.appendingPathComponent("missing.png").path], budget: budget)
            let vector = try #require(results[0].vector, "\(results[0].error ?? "no vector")")
            let expected = try #require(reference.vectors[name])
            let cosine = zip(vector, expected).reduce(Float(0)) { $0 + $1.0 * $1.1 }
            print("Core ML media \(name): cosine=\(cosine)")
            #expect(vector.count == 768 && vector.allSatisfy(\.isFinite))
            #expect(cosine >= 0.995, "\(name) diverged from its reference: \(cosine)")
            #expect(results[1].vector == nil && results[1].error != nil)
        }
        let result = try media.embedAudio([Self.folder.appendingPathComponent("speech.wav").path])[0]
        let vector = try #require(result.vector, "\(result.error ?? "no audio vector")")
        let expected = try #require(reference.vectors["speech"])
        let cosine = zip(vector, expected).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        print("Core ML media speech: cosine=\(cosine)")
        #expect(vector.count == 768 && vector.allSatisfy(\.isFinite))
        #expect(cosine >= 0.98, "Audio FP16 reference agreement: \(cosine)")
        // A model installation/compilation failure stops the indexing run; it must not mark every input failed.
        // This uses the already loaded text encoder but never loads a second media model.
        let missingFolder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: missingFolder) }
        let unavailable = CoreMLMediaEmbedder(folder: missingFolder, text: text)
        #expect(try unavailable.embedImages([], budget: 70).isEmpty)
        #expect(try unavailable.embedAudio([]).isEmpty)
        #expect(throws: WorkerError.self) { try unavailable.embedImages(["missing.png"], budget: 70) }
        #expect(throws: WorkerError.self) { try unavailable.embedAudio(["missing.wav"]) }
        unavailable.close()
        media.close()
        #expect(throws: CoreMLError.self) { try media.embedImages([], budget: 70) }
        #expect(throws: CoreMLError.self) { try media.embedAudio([]) }
    }
}
