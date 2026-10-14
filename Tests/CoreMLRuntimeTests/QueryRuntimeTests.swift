import CoreML
import DigUpKit
import Foundation
import Testing
@testable import CoreMLRuntime

/// Exercise the actual fixed-shape graphs. Run explicitly with the pinned assets and DIGUP_TEST_COREML=1.
@Suite(.enabled(if: ProcessInfo.processInfo.environment["DIGUP_TEST_COREML"] == "1"), .serialized)
struct CoreMLQueryRuntimeTests {
    private func cosine(_ lhs: [Float], _ rhs: [Float]) -> Double {
        zip(lhs, rhs).reduce(0) { $0 + Double($1.0) * Double($1.1) }
    }

    private func checkVector(_ vector: [Float]) {
        #expect(vector.count == 768 && vector.allSatisfy(\.isFinite))
        #expect(abs(vector.reduce(Double(0)) { $0 + Double($1) * Double($1) } - 1) < 0.0001)
    }

    @Test func preparedQuery128MatchesShorterDocumentGraphs() throws {
        let folder = GemmaTokenizerTests.modelFolder
        let query = try CoreMLTextEmbedder(folder: folder, textOnly: true)
        let document = try CoreMLTextEmbedder(folder: folder, textOnly: false)
        defer { query.close(); document.close() }
        try query.prepareForQueries()
        let texts = ["", Prompts.query("payment declined error"), Prompts.query("최근 읽은 인공지능 논문"),
                     Prompts.query("মাছ রান্না"), Prompts.query("找昨天的照片")]
        let queryVectors = try query.embedTexts(texts)
        let documentVectors = try document.embedTexts(texts)
        for (index, pair) in zip(queryVectors, documentVectors).enumerated() {
            checkVector(pair.0)
            checkVector(pair.1)
            let agreement = cosine(pair.0, pair.1)
            print("Core ML query 128 padding agreement [\(index)]: \(agreement)")
            #expect(agreement >= 0.999, "Query padding changed the embedding for case \(index)")
        }
    }

    @Test func query128To512BoundaryAndTruncationKeepTheirTokens() throws {
        let folder = GemmaTokenizerTests.modelFolder
        let tokenizer = try GemmaTokenizer(tokenizerJsonURL: folder.appendingPathComponent("tokenizer.json"))
        let query = try CoreMLTextEmbedder(folder: folder, textOnly: true)
        let document = try CoreMLTextEmbedder(folder: folder, textOnly: false)
        defer { query.close(); document.close() }
        let prefix = Prompts.query("timeline")
        let prefixCount = tokenizer.tokenize(prefix).count
        #expect(prefixCount < 126)
        func text(bodyTokens: Int) -> String {
            // Added tokens give exact boundaries without relying on how BPE merges repeated ordinary words.
            prefix + String(repeating: "<pad>", count: max(0, bodyTokens - prefixCount))
        }
        for bodyCount in [126, 127, 510] {
            let input = text(bodyTokens: bodyCount)
            #expect(tokenizer.tokenize(input).count == bodyCount)
            let actual = try #require(try query.embedTexts([input]).first)
            let expected = try #require(try document.embedTexts([input]).first)
            checkVector(actual)
            let agreement = cosine(actual, expected)
            print("Core ML query total tokens \(bodyCount + 2): agreement=\(agreement)")
            #expect(agreement >= 0.999, "Shape boundary changed tokens or padding at \(bodyCount + 2)")
        }
        let truncated = try #require(try query.embedTexts([text(bodyTokens: 516)]).first)
        let exact = try #require(try query.embedTexts([text(bodyTokens: 510)]).first)
        #expect(zip(truncated, exact).allSatisfy { abs($0 - $1) < 0.000001 })
    }

    @Test func stridedMediaTokensMatchContiguousTokensAndIgnoreInvalidTail() throws {
        let text = try CoreMLTextEmbedder(folder: GemmaTokenizerTests.modelFolder, textOnly: false)
        defer { text.close() }
        let contiguous = try mediaTokens(dataType: .float32, strided: false)
        let paddedFloat = try mediaTokens(dataType: .float32, strided: true)
        let paddedHalf = try mediaTokens(dataType: .float16, strided: true)
        for modality in [CoreMLTextEmbedder.MediaModality.image, .audio] {
            let expected = try text.embedMediaTokens(contiguous, modality: modality, validCount: 3)
            checkVector(expected)
            for tokens in [paddedFloat, paddedHalf] {
                let actual = try text.embedMediaTokens(tokens, modality: modality, validCount: 3)
                checkVector(actual)
                #expect(cosine(actual, expected) >= 0.99999)
            }
        }
        // The fourth row and storage padding contain NaNs. Including it must fail; ignoring it above must succeed.
        #expect(throws: CoreMLError.self) {
            try text.embedMediaTokens(paddedFloat, modality: .image, validCount: 4)
        }
        contiguous.dataPointer.assumingMemoryBound(to: Float.self)[0] = 100_000
        #expect(throws: CoreMLError.self) {
            try text.embedMediaTokens(contiguous, modality: .image, validCount: 3)
        }
    }

    private func mediaTokens(dataType: MLMultiArrayDataType, strided: Bool) throws -> MLMultiArray {
        let rows = 4, columns = 512
        let columnStride = strided ? 2 : 1
        let rowStride = strided ? 1031 : columns
        let count = rows * rowStride
        let bytes = dataType == .float16 ? MemoryLayout<Float16>.stride : MemoryLayout<Float>.stride
        let storage = UnsafeMutableRawPointer.allocate(byteCount: count * bytes, alignment: 64)
        if dataType == .float16 {
            let values = storage.bindMemory(to: Float16.self, capacity: count)
            values.initialize(repeating: .nan, count: count)
            for row in 0..<3 {
                for column in 0..<columns {
                    values[row * rowStride + column * columnStride] = Float16(Float(row * 13 + column % 31 - 15) / 128)
                }
            }
        } else {
            let values = storage.bindMemory(to: Float.self, capacity: count)
            values.initialize(repeating: .nan, count: count)
            for row in 0..<3 {
                for column in 0..<columns {
                    values[row * rowStride + column * columnStride] = Float(row * 13 + column % 31 - 15) / 128
                }
            }
        }
        do {
            return try MLMultiArray(dataPointer: storage, shape: [1, NSNumber(value: rows), NSNumber(value: columns)],
                dataType: dataType, strides: [NSNumber(value: count), NSNumber(value: rowStride), NSNumber(value: columnStride)],
                deallocator: { pointer in pointer.deallocate() })
        } catch {
            storage.deallocate()
            throw error
        }
    }
}
