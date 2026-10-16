import Foundation
import Testing
@testable import NeuralSpotlightKit

@Suite(.serialized) struct ANNTests {
    private func temporary() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("NeuralSpotlightANN-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }
    private func vector(_ seed: UInt64) -> [Float] {
        var state = seed &+ 1
        var result = [Float](repeating: 0, count: Vectors.dimension)
        for index in result.indices {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            result[index] = Float(Int32(truncatingIfNeeded: state >> 32)) / Float(Int32.max)
        }
        let norm = sqrt(result.reduce(Float(0)) { $0 + $1 * $1 })
        return result.map { $0 / norm }
    }
    @discardableResult
    private func add(_ index: Int, store: IndexStore, kind: FileKind = .image, modality: Modality = .image,
                     vector: [Float]) throws -> (file: Int64, segment: Int64) {
        let file = try store.insert(Candidate(path: "/fixtures/asset-\(index).bin", kind: kind, size: 100, mtime: 1,
                                              inode: UInt64(index + 1), device: 1))
        try store.addSegment(file: file, kind: .image, modality: modality, vector: vector)
        return (file, store.db.lastInsertID)
    }

    @Test func cacheReconcilesUpdateDeleteAndReusedIDs() throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try IndexStore(directory: directory)
        try store.setMeta("vector_source", "model-a")
        let target = try add(0, store: store, vector: vector(0))
        for index in 1..<80 { try add(index, store: store, vector: vector(UInt64(index))) }
        #expect(try Searcher(store: store).vectorBackend == "exact")
        let searcher = try Searcher(store: store, vectorSearchMode: .approximate)
        #expect(searcher.vectorBackend == "usearch-hnsw")
        #expect(try searcher.search("", queryVector: vector(0), limit: 1).first?.segmentID == target.segment)
        let restored = try Searcher(store: store, vectorSearchMode: .approximate)
        #expect(restored.annDiagnostic == "cache loaded and reconciled")
        // Reusing the same segment ID with different bytes MUST invalidate its previous graph vector.
        try store.db.run("UPDATE segments SET vec = ? WHERE id = ?", [.blob(Vectors.half(vector(900))), .int(target.segment)])
        try restored.reload()
        #expect(try restored.search("", queryVector: vector(900), limit: 1).first?.segmentID == target.segment)
        try store.delete(target.file)
        try restored.reload()
        #expect(try restored.search("", queryVector: vector(900), limit: 100, zGate: -.infinity).allSatisfy { $0.segmentID != target.segment })
        // Metadata changes also invalidate a loaded cache even with identical vector dimensions/counts.
        try store.setMeta("vector_source", "model-b")
        try restored.reload()
        #expect(restored.vectorSource == "model-b")
        #expect(restored.annDiagnostic == "built from SQLite")
        restored.releaseVectors()
        #expect(!restored.vectorsLoaded)
        #expect(try restored.search("", queryVector: vector(3), limit: 1).first?.path == "/fixtures/asset-3.bin")
    }

    @Test func corruptOrMissingGraphRebuildsFromSQLite() throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try IndexStore(directory: directory)
        for index in 0..<40 { try add(index, store: store, vector: vector(UInt64(index))) }
        _ = try Searcher(store: store, vectorSearchMode: .approximate)
        let cache = directory.appendingPathComponent("index.sqlite.ann")
        let graph = try #require(FileManager.default.contentsOfDirectory(at: cache, includingPropertiesForKeys: nil).first { $0.pathExtension == "usearch" })
        try Data("interrupted write".utf8).write(to: graph)
        let searcher = try Searcher(store: store, vectorSearchMode: .approximate)
        #expect(searcher.vectorBackend == "usearch-hnsw")
        #expect(searcher.annDiagnostic == "built from SQLite")
        #expect(try searcher.search("", queryVector: vector(3), limit: 1).first?.path == "/fixtures/asset-3.bin")
        try FileManager.default.removeItem(at: cache)
        let rebuilt = try Searcher(store: store, vectorSearchMode: .approximate)
        #expect(try rebuilt.search("", queryVector: vector(3), limit: 1).first?.path == "/fixtures/asset-3.bin")
    }

    @Test func calibrationIsIndependentOfNeighborsAndWeightedByPopulation() throws {
        let modalities = Array(repeating: Modality.image, count: 5000) + Array(repeating: .audio, count: 100)
        let sample = MeaningScale.sample(modalities: modalities, setsScale: Array(repeating: true, count: modalities.count), perModality: 64)
        #expect(sample.rows.count == 128)
        #expect(sample.populations == [.image: 5000, .audio: 100])
        #expect(sample.rows == MeaningScale.sample(modalities: modalities, setsScale: Array(repeating: true, count: modalities.count), perModality: 64).rows)
        let allScores = (0..<5000).map { Float($0 % 2 == 0 ? 0.1 : 0.3) } + (0..<100).map { Float($0 % 2 == 0 ? 0.7 : 0.9) }
        let exact = Searcher.meaning(allScores, modalities)
        let fullScale = MeaningScale(scores: allScores, modalities: modalities, populations: sample.populations)
        for row in allScores.indices { #expect(abs(exact[row] - fullScale.score(allScores[row], modality: modalities[row])) < 0.00001) }
        let excluded = MeaningScale.sample(modalities: modalities, setsScale: modalities.indices.map { $0 < 10 })
        #expect(excluded.rows == Array(0..<10))
        #expect(excluded.populations == [.image: 10])
    }

    @Test func concurrentCacheWritersLeaveARecoverableSnapshot() async throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try IndexStore(directory: directory)
        for index in 0..<80 { try add(index, store: store, vector: vector(UInt64(index))) }
        let query = vector(3)
        let successes = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<4 {
                group.addTask {
                    do {
                        let ownStore = try IndexStore(directory: directory)
                        let searcher = try Searcher(store: ownStore, vectorSearchMode: .approximate)
                        return try searcher.search("", queryVector: query, limit: 1).first?.path == "/fixtures/asset-3.bin"
                    } catch { return false }
                }
            }
            var passed = 0
            for await success in group where success { passed += 1 }
            return passed
        }
        #expect(successes == 4)
        let fresh = try Searcher(store: store, vectorSearchMode: .approximate)
        #expect(fresh.annDiagnostic == "cache loaded and reconciled")
        #expect(try fresh.search("", queryVector: query, limit: 1).first?.path == "/fixtures/asset-3.bin")
    }

    @Test func annRecallFiltersGroupingAndExactBaseline() throws {
        let directory = try temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try IndexStore(directory: directory)
        let benchmark = Int((ProcessInfo.processInfo.environment["NEURALSPOTLIGHT_ANN_BENCHMARK"] ?? ProcessInfo.processInfo.environment["DIGUP_ANN_BENCHMARK"]) ?? "0") ?? 0
        let count = benchmark > 1 ? benchmark : benchmark == 1 ? 10000 : 2500
        try store.db.transaction {
            for index in 0..<count {
                try add(index, store: store, kind: index % 20 == 0 ? .pdf : .image,
                        vector: vector(UInt64(index)))
            }
        }
        let builtAt = Date()
        let approximate = try Searcher(store: store, vectorSearchMode: count >= Searcher.annThreshold ? .automatic : .approximate)
        let buildSeconds = Date().timeIntervalSince(builtAt)
        let exact = try Searcher(store: store, vectorSearchMode: .exact)
        #expect(approximate.vectorBackend == "usearch-hnsw")
        #expect(exact.vectorBackend == "exact")
        let firstQueryAt = Date()
        _ = try approximate.search("", queryVector: vector(UInt64(count)), limit: 10, zGate: -.infinity)
        let firstQueryMilliseconds = Date().timeIntervalSince(firstQueryAt) * 1000
        var recovered = 0, annMilliseconds: [Double] = [], exactMilliseconds: [Double] = []
        for query in 0..<10 {
            let embedding = vector(UInt64(count + query * 131))
            var start = Date()
            let reference = try exact.search("", queryVector: embedding, limit: 10, zGate: -.infinity)
            exactMilliseconds.append(Date().timeIntervalSince(start) * 1000)
            start = Date()
            let hits = try approximate.search("", queryVector: embedding, limit: 10, zGate: -.infinity)
            annMilliseconds.append(Date().timeIntervalSince(start) * 1000)
            recovered += Set(hits.map(\.path)).intersection(reference.map(\.path)).count
            #expect(Set(hits.map(\.path)).count == hits.count)
        }
        #expect(recovered >= 98) // deterministic synthetic recall@10; not a claim about real media embeddings.
        let filtered = try approximate.search("", queryVector: vector(0), kinds: [.pdf], limit: 10, zGate: -.infinity)
        let filteredExact = try exact.search("", queryVector: vector(0), kinds: [.pdf], limit: 10, zGate: -.infinity)
        #expect(filtered.map(\.path) == filteredExact.map(\.path))
        #expect(filtered.allSatisfy { $0.kind == .pdf })
        print("ANN benchmark (synthetic, \(count)×768 FP16): build=\(String(format: "%.3f", buildSeconds))s, first ANN query=\(String(format: "%.2f", firstQueryMilliseconds))ms, recall@10=\(recovered)/100, warm exact p95=\(String(format: "%.2f", exactMilliseconds.sorted().last!))ms, warm ANN p95=\(String(format: "%.2f", annMilliseconds.sorted().last!))ms")
        // Many top neighbors from a single long file must not crowd other files out of the result list.
        let crowded = try #require(store.files(under: ["/fixtures"]).first { $0.path == "/fixtures/asset-0.bin" })
        try store.db.transaction {
            for _ in 0..<350 { try store.addSegment(file: crowded.id, kind: .image, modality: .image, vector: vector(0)) }
        }
        try approximate.reload()
        let grouped = try approximate.search("", queryVector: vector(0), limit: 10, zGate: -.infinity)
        #expect(grouped.count == 10)
        #expect(Set(grouped.map(\.path)).count == 10)
        #expect(grouped.first?.path == crowded.path)
    }
}
