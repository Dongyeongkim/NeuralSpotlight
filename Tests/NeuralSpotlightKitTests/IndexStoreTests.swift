#if canImport(CryptoKit)
import CryptoKit
#endif
import Foundation
import Testing
@testable import NeuralSpotlightKit
#if os(Windows)
import WinSDK
#endif

@Suite struct IndexStoreTests {
    let folder: URL = {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("NeuralSpotlightTests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }()

    func file(_ path: String, inode: UInt64) -> Candidate {
        Candidate(path: path, kind: .image, size: 10, mtime: 1, inode: inode, device: 1)
    }

    @Test func pruneKeepsOnlyChosenFolders() throws {
        let store = try IndexStore(directory: folder)
        try store.insert(file("/a/x.png", inode: 1))
        try store.insert(file("/a/sub/y.png", inode: 2))
        try store.insert(file("/ab/z.png", inode: 3))   // a sibling whose name starts like "/a", not inside it
        try store.insert(file("/b/w.png", inode: 4))
        #expect(try store.prune(keepingUnder: ["/a"]) == 2)
        #expect(try store.files(under: ["/a"]).count == 2)
        #expect(try store.files(under: ["/ab", "/b"]).isEmpty)
        var keywordRows = 0
        try store.db.query("SELECT COUNT(*) FROM fts") { keywordRows = Int($0.int(0)) }
        #expect(keywordRows == 2)   // the pruned files' names left keyword search too
        #expect(try store.prune(keepingUnder: []) == 2)
    }

    @Test func scopeLimitsWhatIsWaiting() throws {
        let store = try IndexStore(directory: folder)
        try store.insert(file("/a/x.png", inode: 1))
        try store.insert(file("/drive/y.png", inode: 2))   // on a drive that's unplugged
        store.scope = ["/a"]
        #expect(try store.pendingCount() == 1)
        #expect(try store.pending(limit: nil).map(\.path) == ["/a/x.png"])
        #expect(try store.counts().byKindAndState["image"]?["pending"] == 1)
        store.scope = []
        #expect(try store.pendingCount() == 0)
        store.scope = nil
        #expect(try store.pendingCount() == 2)
    }

    @Test func aFileTheHelperStoppedOnIsNotReadAgain() throws {
        let store = try IndexStore(directory: folder)
        try store.insert(file("/a/stuck.pdf", inode: 1))
        let book = try store.insert(file("/a/book.pdf", inode: 2))
        try store.finish(book, state: "partial", info: ["read_to": "30", "read_end": "463"])
        #expect(try store.markStopped("/a/stuck.pdf", why: "took more than 15 minutes to read"))
        #expect(try store.markStopped("/a/book.pdf", why: "took more than 15 minutes to read"))
        #expect(try !store.markStopped("/a/gone.pdf", why: "took more than 15 minutes to read"))
        #expect(try store.pendingCount() == 0)
        #expect(try store.unfinishedCount().files == 0)   // the book keeps its first pages and isn't finished again
        let states = Dictionary(uniqueKeysWithValues: try store.files(under: ["/a"]).map { ($0.path, $0.state) })
        #expect(states == ["/a/stuck.pdf": "failed", "/a/book.pdf": "done"])
    }

    #if canImport(AppKit)
    @Test func oldMediaDecodeFailuresAreRecheckedOnceWithoutTouchingOtherFiles() throws {
        let store = try IndexStore(directory: folder)
        func add(_ name: String, kind: FileKind, inode: UInt64, state: String, detail: String?) throws -> Int64 {
            let id = try store.insert(Candidate(path: "/a/\(name)", kind: kind, size: 10, mtime: 1,
                                                inode: inode, device: 1))
            try store.finish(id, state: state, detail: detail)
            return id
        }
        let decoderError = "Error Domain=AVFoundationErrorDomain Code=-11821 Cannot Decode"
        let movie = try add("protected.m4v", kind: .video, inode: 1, state: "failed", detail: decoderError)
        _ = try add("broken.pdf", kind: .pdf, inode: 2, state: "failed", detail: "unreadable: can't open PDF")
        _ = try add("slow.mov", kind: .video, inode: 3, state: "failed", detail: "It took too long to read")
        _ = try add("ready.mp4", kind: .video, inode: 4, state: "done", detail: nil)

        #expect(try store.readAgainOnce() == 1)
        #expect(try store.pending(limit: nil).map(\.id) == [movie])
        // Classification happens after checking the actual asset, not by guessing from the decoder error.
        try store.finish(movie, state: "unsupported", detail: "unsupported: Protected media (DRM)")
        #expect(try store.readAgainOnce() == 0)
        #expect(try store.pendingCount() == 0)
        let counts = try store.counts().byKindAndState
        #expect(counts["video"]?["unsupported"] == 1)
        #expect(counts["video"]?["failed"] == 1)
        #expect(counts["video"]?["done"] == 1)
        #expect(counts["pdf"]?["failed"] == 1)
        let searcher = try Searcher(store: store)
        #expect(try searcher.search("protected", queryVector: nil).contains { $0.path == "/a/protected.m4v" })
    }

    @Test func picturesThatCouldntBeReadAreReadOnceMore() throws {
        // An SVG skipped by an older DigUp as "small image 0×0" (ImageIO couldn't open it): AppKit draws it now.
        let store = try IndexStore(directory: folder)
        let svg = try store.insert(file("/a/chart.svg", inode: 1))
        try store.finish(svg, state: "skipped", detail: "small image 0×0")
        let icon = try store.insert(file("/a/icon.png", inode: 2))
        try store.finish(icon, state: "skipped", detail: "small image 160×160")   // small, and rightly skipped
        #expect(try store.readAgainOnce() == 1)
        #expect(try store.pending(limit: nil).map(\.path) == ["/a/chart.svg"])
        #expect(try store.readAgainOnce() == 0)
    }
    #endif

    @Test func modelFileHashesStreamInPieces() throws {
        let small = folder.appendingPathComponent("abc")
        try Data("abc".utf8).write(to: small)
        #expect(try ModelFiles.sha256(of: small) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        // Bigger than one 8 MB piece.
        let bytes = Data((0..<(9 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })
        let big = folder.appendingPathComponent("big")
        try bytes.write(to: big)
        let expected = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        #expect(try ModelFiles.sha256(of: big) == expected)
    }

    @Test func missingModelFilesInDownloadOrder() throws {
        let files = CoreMLModelFiles.all
        #expect(CoreMLModelFiles.missing(in: folder) == files)
        let first = folder.appendingPathComponent(files[0].name)
        try FileManager.default.createDirectory(at: first.deletingLastPathComponent(), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: first.path, contents: nil)
        let handle = try FileHandle(forWritingTo: first)
        try handle.truncate(atOffset: UInt64(files[0].bytes))
        try handle.close()
        #expect(CoreMLModelFiles.missing(in: folder) == Array(files.dropFirst()))
        #expect(CoreMLModelFiles.totalBytes == files.reduce(0) { $0 + $1.bytes })
    }

    @Test func modelFilesComeFromTheHuggingFaceCacheWhenThere() throws {
        // As `hf download` leaves them: snapshots/<revision>/<name> → a blob. Sparse, so no disk is used.
        let hub = folder.appendingPathComponent("hub")
        let file = ModelFiles.File(name: "CoreML/tokenizer.json", bytes: 1024, sha256: "fixture",
                                   repository: "fixture/model", revision: "pinned", remotePath: "tokenizer.json")
        let absent = ModelFiles.File(name: "CoreML/weights.bin", bytes: 2048, sha256: "other",
                                     repository: file.repository, revision: file.revision)
        let snapshot = hub.appendingPathComponent("models--fixture--model/snapshots/pinned")
        let blobs = hub.appendingPathComponent("models--fixture--model/blobs")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let blob = blobs.appendingPathComponent("fixture-blob")
        FileManager.default.createFile(atPath: blob.path, contents: nil)
        try FileHandle(forWritingTo: blob).truncate(atOffset: UInt64(file.bytes))
        try FileManager.default.createSymbolicLink(at: snapshot.appendingPathComponent(file.remotePath),
                                                   withDestinationURL: blob)
        setEnvironment("HF_HUB_CACHE", hub.path)
        defer { setEnvironment("HF_HUB_CACHE", nil) }
        #expect(ModelFiles.cachedCopy(of: file)?.lastPathComponent == "fixture-blob")
        #expect(ModelFiles.cachedCopy(of: absent) == nil)   // not there
        try FileHandle(forWritingTo: blob).truncate(atOffset: 10)
        #expect(ModelFiles.cachedCopy(of: file) == nil)    // not whole
    }

    @Test func syncKeepsTheRowsOfAFolderThatWentAway() throws {
        // The real path (/private/var/…), as the crawler reports it; resolvingSymlinksInPath() would drop "/private".
        let base = URL(fileURLWithPath: FolderSelection.realPath(folder.path))
        let drive = base.appendingPathComponent("drive")
        try FileManager.default.createDirectory(at: drive, withIntermediateDirectories: true)
        try Data("notes from the trip".utf8).write(to: drive.appendingPathComponent("notes.txt"))
        let store = try IndexStore(directory: base.appendingPathComponent("index"))
        let indexer = try Indexer(store: store, options: IndexOptions(),
                                  makeEmbedder: { throw WorkerError.notFound("no model needed to sync") })
        #expect(try indexer.sync([drive]).added == 1)
        try FileManager.default.moveItem(at: drive, to: base.appendingPathComponent("ejected"))
        #expect(try indexer.sync([drive]).removed == 0)   // an ejected drive is not "everything deleted"
        #expect(try store.files(under: [drive.path]).count == 1)
    }

    #if !os(Windows)
    @Test func unreadableSubfolderKeepsItsIndexedVectors() throws {
        let base = URL(fileURLWithPath: FolderSelection.realPath(folder.path))
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appendingPathComponent("files")
        let locked = root.appendingPathComponent("locked")
        try FileManager.default.createDirectory(at: locked, withIntermediateDirectories: true)
        let document = locked.appendingPathComponent("notes.txt")
        try Data("Important searchable notes".utf8).write(to: document)
        let store = try IndexStore(directory: base.appendingPathComponent("index"))
        let options = IndexOptions()
        let indexer = try Indexer(store: store, options: options,
                                  makeEmbedder: { throw WorkerError.notFound("no model needed to sync") })
        #expect(try indexer.sync([root]).added == 1)
        let row = try #require(store.files(under: [root.path]).first)
        try store.addSegment(file: row.id, kind: .chunk, modality: .text,
                             text: "Important searchable notes", vector: [Float](repeating: 0.1, count: 768))
        try FileManager.default.setAttributes([.posixPermissions: 0], ofItemAtPath: locked.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: locked.path) }
        let report = Crawler.crawl([root], options: options)
        #expect(report.unreadablePaths.contains(locked.path))
        #expect(try indexer.sync([root]).removed == 0)
        #expect(try store.files(under: [root.path]).count == 1)
        #expect(try store.counts().vectors == 1)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: locked.path)
        try FileManager.default.removeItem(at: document)
        #expect(try indexer.sync([root]).removed == 1)
        #expect(try store.counts().vectors == 0)
    }
    #endif

    @Test(arguments: [false, true])
    func deletingFilesRemovesTextVectorsActivityAndANNCandidates(code: Bool) throws {
        let base = URL(fileURLWithPath: FolderSelection.realPath(folder.path))
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appendingPathComponent("files")
        let subtree = root.appendingPathComponent("nested")
        try FileManager.default.createDirectory(at: subtree, withIntermediateDirectories: true)
        let ext = code ? "swift" : "txt"
        let gone = subtree.appendingPathComponent("deletedfixture.\(ext)")
        let kept = root.appendingPathComponent("retainedfixture.\(ext)")
        for url in [gone, kept] { try Data("let fixture = 1\n".utf8).write(to: url) }
        var options = IndexOptions()
        options.code = code
        let store = try IndexStore(directory: base.appendingPathComponent("index"),
                                   file: code ? IndexStore.codeFile : IndexStore.mainFile)
        let indexer = try Indexer(store: store, options: options,
                                  makeEmbedder: { throw WorkerError.notFound("deletion must not load a model") })
        #expect(try indexer.sync([root]).added == 2)
        let files = try store.files(under: [root.path])
        var vector = [Float](repeating: 0, count: Vectors.dimension)
        vector[0] = 1
        for file in files {
            try store.addSegment(file: file.id, kind: .chunk, modality: .text,
                                 text: "deletionfixturecontent", vector: vector)
            try store.finish(file.id, state: "done")
            try store.recordUse(path: file.path, at: Date(), source: "open")
        }
        let searcher = try Searcher(store: store, vectorSearchMode: .approximate)
        #expect(searcher.vectorCount == 2)
        // Removing a directory must remove all its indexed descendants, but not its sibling.
        try FileManager.default.removeItem(at: subtree)
        #expect(try indexer.sync([root]).removed == 1)
        #expect(try store.files(under: [root.path]).map(\.path) == [kept.path])
        var activity: [String] = []
        try store.db.query("SELECT path FROM activity") { activity.append($0.text(0)!) }
        #expect(activity == [kept.path])
        searcher.releaseVectors() // the app invalidates the loaded search cache on a deletion sync
        #expect(try searcher.search("deletedfixture", queryVector: nil).isEmpty)
        let hits = try searcher.search("", queryVector: vector, limit: 10, zGate: -.infinity)
        #expect(hits.map(\.path) == [kept.path])
        #expect(searcher.vectorCount == 1)
        #expect(try Searcher(store: store, vectorSearchMode: .approximate)
            .search("", queryVector: vector, limit: 10, zGate: -.infinity).map(\.path) == [kept.path])
        #expect(try indexer.sync([root]).removed == 0)
    }

    @Test func syncFindsTheRowsOfAFolderSpelledAnotherWay() throws {
        // The walk stores paths as the disk spells them, and on a Mac these folders are chosen spelled otherwise twice
        // over: under /var where the disk says /private/var (the temporary folder), and with a name the disk keeps
        // precomposed that `standardizedFileURL` decomposes (Google Drive's "내 드라이브", "Café"). Looking their rows up
        // by that spelling found none, so the second sync took every file for new: "UNIQUE constraint failed".
        let store = try IndexStore(directory: folder.appendingPathComponent("index"))
        let indexer = try Indexer(store: store, options: IndexOptions(),
                                  makeEmbedder: { throw WorkerError.notFound("no model needed to sync") })
        var chosen: [URL] = []
        for name in ["내 드라이브", "Café"] {
            let onDisk = folder.path + "/" + name.precomposedStringWithCanonicalMapping
            #if os(Windows)
            try FileManager.default.createDirectory(atPath: onDisk, withIntermediateDirectories: false)
            #else
            #expect(mkdir(onDisk, 0o755) == 0)   // FileManager would decompose the name
            #endif
            try Data("notes from the trip".utf8).write(to: URL(fileURLWithPath: onDisk + "/notes.txt"))
            chosen.append(URL(fileURLWithPath: onDisk).standardizedFileURL)
        }
        #expect(try indexer.sync(chosen).added == 2)
        let again = try indexer.sync(chosen)
        #expect(again.added == 0 && again.unchanged == 2)
        #expect(try store.files(under: chosen.map(\.path)).count == 2)
        store.scope = chosen.map(\.path)
        #expect(try store.pendingCount() == 2)
        #expect(try store.prune(keepingUnder: chosen.map(\.path)) == 0)
    }

    @Test func targetedSyncAndContentDigestSkipUnchangedMtimeUpdates() throws {
        let base = URL(fileURLWithPath: FolderSelection.realPath(folder.path))
        defer { try? FileManager.default.removeItem(at: base) }
        let root = base.appendingPathComponent("files")
        let dirA = root.appendingPathComponent("a")
        let dirB = root.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: dirA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dirB, withIntermediateDirectories: true)
        let fileA = dirA.appendingPathComponent("alpha.txt")
        let fileB = dirB.appendingPathComponent("beta.txt")
        try Data("Alpha content for testing".utf8).write(to: fileA)
        try Data("Beta content for testing".utf8).write(to: fileB)
        let store = try IndexStore(directory: base.appendingPathComponent("index"))
        let embedder = FakeEmbedder()
        let indexer = try Indexer(store: store, options: IndexOptions(), makeEmbedder: { embedder })
        #expect(try indexer.sync([root]).added == 2)
        _ = try indexer.run()
        #expect(embedder.texts == 2)

        // Touching fileA's mtime without changing its bytes must not re-queue it as pending.
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(120)], ofItemAtPath: fileA.path)
        let touched = try indexer.sync(paths: [fileA.path], roots: [root])
        #expect(touched.changed == 0 && touched.unchanged == 1 && touched.queued.isEmpty)
        #expect(try store.pendingCount() == 0)

        // Adding a new file and modifying fileA via targeted sync updates only dirA without touching dirB.
        let fileA2 = dirA.appendingPathComponent("added.txt")
        try Data("Newly added file in dirA".utf8).write(to: fileA2)
        try store.recordUse(path: fileA2.path, at: Date(), source: "open")
        #expect(try store.unindexedActivityPaths() == [fileA2.path])
        let added = try indexer.sync(paths: [fileA2.path], roots: [root])
        #expect(added.added == 1 && added.queued.count == 1)
        #expect(try store.unindexedActivityPaths().isEmpty)

        // Deleting fileA2 via targeted sync removes only fileA2 and keeps fileB untouched.
        try FileManager.default.removeItem(at: fileA2)
        let removed = try indexer.sync(paths: [fileA2.path], roots: [root])
        #expect(removed.removed == 1)
        #expect(Set(try store.files(under: [root.path]).map(\.path)) == [fileA.path, fileB.path])
    }
}

/// Sets an environment variable of this process, or removes it (nil).
private func setEnvironment(_ name: String, _ value: String?) {
    #if os(Windows)
    _ = name.withCString(encodedAs: UTF16.self) { name in
        guard let value else { return SetEnvironmentVariableW(name, nil) }
        return value.withCString(encodedAs: UTF16.self) { SetEnvironmentVariableW(name, $0) }
    }
    #else
    if let value { setenv(name, value, 1) } else { unsetenv(name) }
    #endif
}
