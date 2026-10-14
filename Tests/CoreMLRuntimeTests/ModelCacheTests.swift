import CryptoKit
import Darwin
import DigUpKit
import Foundation
import Testing
@testable import CoreMLRuntime

@Suite struct ModelCacheTests {
    private func fixture() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    private func asset(_ text: String, path: String = "weights.bin", revision: String = "fixture-v1") -> ModelFiles.File {
        let data = Data(text.utf8)
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return ModelFiles.File(name: "CoreML/\(path)", bytes: Int64(data.count), sha256: hash,
                               repository: "tests/model", revision: revision, remotePath: path)
    }

    private func receipts(in folder: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("Compiled"),
            includingPropertiesForKeys: nil).filter { $0.lastPathComponent.hasPrefix("verified-") && $0.pathExtension == "json" }
    }

    @Test func unchangedFilesReuseReceiptAndManifestOrderDoesNotMatter() throws {
        let folder = try fixture()
        defer { try? FileManager.default.removeItem(at: folder) }
        let files = [asset("hello"), asset("world", path: "tokenizer.json")]
        try Data("hello".utf8).write(to: folder.appendingPathComponent("weights.bin"))
        try Data("world".utf8).write(to: folder.appendingPathComponent("tokenizer.json"))
        try CoreMLModelCache.verify(folder: folder, files: files)
        let receipt = try #require(try receipts(in: folder).first)
        let before = try FileManager.default.attributesOfItem(atPath: receipt.path)
        try CoreMLModelCache.verify(folder: folder, files: Array(files.reversed()))
        let after = try FileManager.default.attributesOfItem(atPath: receipt.path)
        #expect(before[.systemFileNumber] as? NSNumber == after[.systemFileNumber] as? NSNumber)
        #expect(before[.modificationDate] as? Date == after[.modificationDate] as? Date)
        #expect(try receipts(in: folder).count == 1)
    }

    @Test func sameSizeEditWithRestoredModificationTimeIsReverified() throws {
        let folder = try fixture()
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("weights.bin")
        try Data("hello".utf8).write(to: file)
        let expected = [asset("hello")]
        try CoreMLModelCache.verify(folder: folder, files: expected)
        var original = stat()
        #expect(stat(file.path, &original) == 0)
        let handle = try FileHandle(forWritingTo: file)
        try handle.write(contentsOf: Data("wrong".utf8))
        try handle.close()
        let times = [original.st_atimespec, original.st_mtimespec]
        let restored = times.withUnsafeBufferPointer { utimensat(AT_FDCWD, file.path, $0.baseAddress, 0) }
        #expect(restored == 0)
        var changed = stat()
        #expect(stat(file.path, &changed) == 0)
        #expect(changed.st_ino == original.st_ino)
        #expect(changed.st_mtimespec.tv_sec == original.st_mtimespec.tv_sec)
        #expect(changed.st_mtimespec.tv_nsec == original.st_mtimespec.tv_nsec)
        #expect(throws: (any Error).self) { try CoreMLModelCache.verify(folder: folder, files: expected) }
    }

    @Test func replacementDeletionAndAddedRequiredFileInvalidateReceipt() throws {
        let folder = try fixture()
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("weights.bin")
        let expected = asset("hello")
        try Data("hello".utf8).write(to: file)
        try CoreMLModelCache.verify(folder: folder, files: [expected])
        try Data("wrong".utf8).write(to: file, options: .atomic)
        #expect(throws: (any Error).self) { try CoreMLModelCache.verify(folder: folder, files: [expected]) }
        try FileManager.default.removeItem(at: file)
        #expect(throws: (any Error).self) { try CoreMLModelCache.verify(folder: folder, files: [expected]) }
        try Data("hello".utf8).write(to: file)
        try CoreMLModelCache.verify(folder: folder, files: [expected])
        #expect(throws: (any Error).self) {
            try CoreMLModelCache.verify(folder: folder, files: [expected, asset("new", path: "added.bin")])
        }
        try Data("new".utf8).write(to: folder.appendingPathComponent("added.bin"))
        try CoreMLModelCache.verify(folder: folder, files: [expected, asset("new", path: "added.bin")])
        #expect(try receipts(in: folder).count == 2)
    }

    @Test func expectedHashAndRevisionArePartOfReceiptIdentity() throws {
        let folder = try fixture()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data("hello".utf8).write(to: folder.appendingPathComponent("weights.bin"))
        try CoreMLModelCache.verify(folder: folder, files: [asset("hello")])
        #expect(throws: (any Error).self) {
            try CoreMLModelCache.verify(folder: folder, files: [asset("wrong")])
        }
        try CoreMLModelCache.verify(folder: folder, files: [asset("hello", revision: "fixture-v2")])
        #expect(try receipts(in: folder).count == 2)
    }

    @Test func corruptReceiptIsRebuiltAfterHashVerification() throws {
        let folder = try fixture()
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data("hello".utf8).write(to: folder.appendingPathComponent("weights.bin"))
        try CoreMLModelCache.verify(folder: folder, files: [asset("hello")])
        let receipt = try #require(try receipts(in: folder).first)
        try Data("incomplete JSON".utf8).write(to: receipt)
        try CoreMLModelCache.verify(folder: folder, files: [asset("hello")])
        #expect((try? JSONSerialization.jsonObject(with: Data(contentsOf: receipt))) != nil)
    }
}
