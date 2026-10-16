import Foundation
import Testing
@testable import NeuralSpotlightKit

@Suite final class ActivityTests {
    let root: URL = {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("NeuralSpotlightActivity-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return URL(fileURLWithPath: FolderSelection.realPath(folder.path))
    }()
    deinit { try? FileManager.default.removeItem(at: root) }
    func store() throws -> IndexStore { try IndexStore(directory: root.appendingPathComponent("index")) }
    func candidate(_ path: String, kind: FileKind = .pdf, inode: UInt64 = 1) -> Candidate {
        Candidate(path: path, kind: kind, size: 1, mtime: 1, inode: inode, device: 1)
    }

    @Test func engagementArrivesBeforeCrawlAndSurvivesReopen() throws {
        let store = try store()
        let now = Date()
        #expect(try store.recordUse(path: "/D/old.pdf", at: now, source: "spotlight", now: now))
        try store.insert(candidate("/D/new.png", kind: .image))
        let engaged = try store.insert(candidate("/D/old.pdf", inode: 2))
        let reopened = try self.store()
        reopened.prioritizeRecent = true
        #expect(try reopened.pending(limit: nil).first?.id == engaged)
        #expect(try reopened.recentFiles().map(\.id) == [engaged])
        #expect(!(try reopened.recordUse(path: "/D/old.pdf", at: now, source: "spotlight", now: now)))
        #expect(!(try reopened.recordUse(path: "/D/old.pdf", at: now.addingTimeInterval(-10), source: "spotlight", now: now)))
    }

    @Test func completedFilesAreNotReembeddedAndScopeIsRespected() throws {
        let store = try store()
        let a = try store.insert(candidate("/D/a.pdf"))
        let b = try store.insert(candidate("/E/b.pdf", inode: 2))
        try store.recordUse(path: "/D/a.pdf", at: Date(), source: "open")
        try store.recordUse(path: "/E/b.pdf", at: Date(), source: "open")
        try store.finish(a, state: "done")
        #expect(try store.recentFiles().map(\.id) == [b])
        store.scope = ["/D"]
        #expect(try store.recentFiles().isEmpty)
        try store.requeue(a)
        #expect(try store.recentFiles().map(\.id) == [a])
    }

    @Test func activityFollowsRenameAndIsRemovedWithFile() throws {
        let store = try store()
        let id = try store.insert(candidate("/D/a.pdf"))
        try store.recordUse(path: "/D/a.pdf", at: Date(), source: "preview")
        try store.move(id, to: "/D/b.pdf")
        #expect(try store.recentFiles().map(\.path) == ["/D/b.pdf"])
        try store.delete(id)
        var count: Int64 = -1
        try store.db.query("SELECT COUNT(*) FROM activity") { count = $0.int(0) }
        #expect(count == 0)
    }

    @Test func unchosenAndExpiredActivityDoesNotRemain() throws {
        let store = try store()
        let now = Date()
        try store.recordUse(path: "/D/a.pdf", at: now, source: "open", now: now)
        try store.recordUse(path: "/E/b.pdf", at: now, source: "open", now: now)
        try store.prune(keepingUnder: ["/D"])
        var paths: [String] = []
        try store.db.query("SELECT path FROM activity") { paths.append($0.text(0)!) }
        #expect(paths == ["/D/a.pdf"])
        try store.expireActivity(now: now.addingTimeInterval(31 * 86400))
        var count: Int64 = -1
        try store.db.query("SELECT COUNT(*) FROM activity") { count = $0.int(0) }
        #expect(count == 0)
        #expect(!(try store.recordUse(path: "/D/a.pdf", at: now.addingTimeInterval(1000), source: "open", now: now)))
    }

    @Test func schemaFiveMigratesWithoutDroppingContent() throws {
        let store = try store()
        let id = try store.insert(candidate("/D/keep.pdf"))
        try store.finish(id, state: "done")
        try store.setMeta("vector_source", "unchanged")
        try store.db.execute("DROP TABLE activity; PRAGMA user_version = 5;")
        let migrated = try self.store()
        #expect(try migrated.meta("vector_source") == "unchanged")
        #expect(try migrated.files(under: ["/D"]).first?.state == "done")
        #expect(try migrated.recordUse(path: "/D/keep.pdf", at: Date(), source: "open"))
    }

    @Test func richPolicyLeavesOrdinaryDocumentsToSpotlight() throws {
        let folder = root.appendingPathComponent("files")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("a plain note".utf8).write(to: folder.appendingPathComponent("note.txt"))
        try TextPDF.write(["A paper"], to: folder.appendingPathComponent("paper.pdf"))
        var options = IndexOptions()
        options.richContentOnly = true
        #expect(Crawler.crawl([folder], options: options).candidates.map(\.kind) == [.pdf])
        options.richContentOnly = false
        #expect(Set(Crawler.crawl([folder], options: options).candidates.map(\.kind)) == [.pdf, .doc])
        #expect(SemanticPolicy.embeds(.code))
        #expect(SemanticPolicy.embeds(.audio))
        #expect(!SemanticPolicy.embeds(.application))
    }

    @Test func onlyEngagedLongFileFinishesAheadOfBackfill() throws {
        let folder = root.appendingPathComponent("files")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for name in ["active.pdf", "later.pdf"] {
            try TextPDF.write(Array(repeating: String(repeating: "Research into memory and learning. ", count: 12), count: 3),
                              to: folder.appendingPathComponent(name))
        }
        var options = IndexOptions()
        options.bigPDFPages = 1
        options.pdfStep = 1
        let store = try store()
        let embedder = FakeEmbedder()
        let indexer = try Indexer(store: store, options: options, makeEmbedder: { embedder })
        _ = try indexer.sync([folder])
        try indexer.run()
        let active = try #require(store.files(under: [folder.path]).first { $0.path.hasSuffix("active.pdf") })
        try store.recordUse(path: active.path, at: Date(), source: "open")
        #expect(try store.recentFiles().first?.state == "partial")
        try indexer.finishLongFiles(only: [active.id])
        #expect(try store.unfinished().map { URL(fileURLWithPath: $0.path).lastPathComponent } == ["later.pdf"])
        #expect(embedder.texts == 4)
    }
}
