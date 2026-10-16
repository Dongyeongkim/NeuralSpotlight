import Foundation
import Testing
@testable import NeuralSpotlightKit

struct RuntimeMigrationTests {
    private func fixture(_ directory: URL) throws -> IndexStore {
        let store = try IndexStore(directory: directory)
        let id = try store.insert(Candidate(path: "/fixture/paper.pdf", kind: .pdf, size: 100,
                                            mtime: 1, inode: 1, device: 1))
        try store.addSegment(file: id, kind: .page, modality: .text, loc: 1,
                             text: "previous searchable content", vector: FakeEmbedder.vector(for: "old model"))
        try store.finish(id, state: "done")
        try store.setMeta("fingerprint", "old-runtime | old-options")
        try store.setMeta("vector_source", "old-runtime")
        try store.recordUse(path: "/fixture/paper.pdf", at: Date(), source: "fixture")
        return store
    }

    @Test func invalidReplacementPreservesExistingSemanticIndex() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try fixture(directory)
        let indexer = try Indexer(store: store, options: IndexOptions()) {
            throw WorkerError.notFound("replacement model checksum failed")
        }
        indexer.embedderSource = "new-runtime"
        var reset = false
        indexer.resetOnModelChange = { _, _ in reset = true }
        #expect(throws: WorkerError.self) { try indexer.checkModel() }
        #expect(!reset)
        #expect(try store.counts().vectors == 1)
        #expect(try store.pendingCount() == 0)
        #expect(try store.meta("vector_source") == "old-runtime")
        #expect(try store.meta("fingerprint") == "old-runtime | old-options")
    }

    @Test func validatedReplacementRequeuesWithoutLosingNamesOrActivity() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = try fixture(directory)
        let replacement = FakeEmbedder()
        var loaded = false
        let indexer = try Indexer(store: store, options: IndexOptions()) {
            loaded = true
            return replacement
        }
        indexer.embedderSource = replacement.info.vectorSource
        var reset = false
        indexer.resetOnModelChange = { _, _ in
            #expect(loaded)
            reset = true
        }
        try indexer.checkModel()
        #expect(reset)
        #expect(try store.counts().vectors == 0)
        #expect(try store.pendingCount() == 1)
        #expect(try store.meta("vector_source") == replacement.info.vectorSource)
        #expect(try store.meta("fingerprint")?.hasPrefix(replacement.info.vectorSource + " | ") == true)
        var names = 0, activity = 0
        try store.db.query("SELECT count(*) FROM segments WHERE kind = 'name'") { names = Int($0.int(0)) }
        try store.db.query("SELECT count(*) FROM activity") { activity = Int($0.int(0)) }
        #expect(names == 1)
        #expect(activity == 1)
        indexer.finish()
    }
}
