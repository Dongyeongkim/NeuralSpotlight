import CoreML
import CryptoKit
import Darwin
import DigUpKit
import Foundation

/// Shared by the text and media runtimes. Receipts avoid rereading unchanged pinned weights when a helper restarts.
/// Compiled models keep a stable path, which Core ML uses with the configuration for its device-specialization cache.
enum CoreMLModelCache {
    private struct ExpectedFile: Codable, Equatable {
        let path: String
        let bytes: Int64
        let sha256: String
        let repository: String
        let revision: String
    }

    private struct VerifiedFile: Codable, Equatable {
        let expected: ExpectedFile
        let device: Int64
        let inode: UInt64
        let size: Int64
        let modifiedSeconds: Int64
        let modifiedNanoseconds: Int64
        let changedSeconds: Int64
        let changedNanoseconds: Int64
    }

    private struct Receipt: Codable, Equatable {
        let version: Int
        let files: [VerifiedFile]
    }

    static func verify(folder: URL, files: [ModelFiles.File]) throws {
        let expected = files.map {
            ExpectedFile(path: $0.remotePath, bytes: $0.bytes, sha256: $0.sha256,
                         repository: $0.repository, revision: $0.revision)
        }.sorted { $0.path < $1.path }
        guard !expected.isEmpty, Set(expected.map(\.path)).count == expected.count else {
            throw CoreMLError.invalidAsset("empty or duplicate model file manifest")
        }
        // Different subsets (text, image, audio) have separate receipts and locks, but every expected hash and
        // revision participates in the identity. Adding a required file cannot reuse an older, smaller receipt.
        let cache = try cacheDirectory(folder)
        let manifest = try encoded(expected)
        let key = SHA256.hash(data: manifest).map { String(format: "%02x", $0) }.joined()
        let receiptURL = cache.appendingPathComponent("verified-\(key).json")
        try withLock(at: cache.appendingPathComponent("verify-\(key).lock")) {
            let before = try Receipt(version: 1, files: expected.map { try identity(of: $0, in: folder) })
            if let data = try? Data(contentsOf: receiptURL),
               let saved = try? JSONDecoder().decode(Receipt.self, from: data), saved == before { return }

            for file in expected {
                let url = try sourceURL(file.path, in: folder)
                guard try ModelFiles.sha256(of: url) == file.sha256 else {
                    throw CoreMLError.invalidAsset("checksum mismatch: \(file.path)")
                }
            }
            let after = try Receipt(version: 1, files: expected.map { try identity(of: $0, in: folder) })
            guard before == after else { throw CoreMLError.invalidAsset("model files changed during verification") }
            try encoded(after).write(to: receiptURL, options: .atomic)
        }
    }

    /// Compile only once per package/revision. Each function and compute-unit configuration can then reuse this
    /// same compiled URL; Core ML owns the separate device-specific cache. Call `verify` before using this method.
    static func compile(folder: URL, package: String) throws -> URL {
        let source = try sourceURL(package, in: folder)
        let cache = try cacheDirectory(folder)
        // Package names supplied by our manifest are single components, avoiding collisions between nested names.
        guard package == source.lastPathComponent else {
            throw CoreMLError.invalidAsset("model package must be directly inside the model folder")
        }
        let name = source.deletingPathExtension().lastPathComponent + "-" + CoreMLModelFiles.revision
        let destination = cache.appendingPathComponent(name + ".mlmodelc", isDirectory: true)
        let marker = "1\n\(package)\n\(CoreMLModelFiles.revision)\n"
        return try withLock(at: cache.appendingPathComponent(name + ".lock")) {
            let stamp = destination.appendingPathComponent("digup-revision")
            if (try? String(contentsOf: stamp, encoding: .utf8)) == marker { return destination }

            let temporary = try MLModel.compileModel(at: source)
            defer { try? FileManager.default.removeItem(at: temporary) }
            let staged = cache.appendingPathComponent(UUID().uuidString + ".mlmodelc", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: staged) }
            try FileManager.default.copyItem(at: temporary, to: staged)
            try marker.write(to: staged.appendingPathComponent("digup-revision"), atomically: true, encoding: .utf8)
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: staged, to: destination)
            return destination
        }
    }

    private static func sourceURL(_ path: String, in folder: URL) throws -> URL {
        let base = folder.standardizedFileURL
        let url = base.appendingPathComponent(path).standardizedFileURL
        let prefix = base.path.hasSuffix("/") ? base.path : base.path + "/"
        guard !path.hasPrefix("/"), url.path.hasPrefix(prefix), url.path != base.path else {
            throw CoreMLError.invalidAsset("model file path escapes its folder")
        }
        return url
    }

    private static func identity(of file: ExpectedFile, in folder: URL) throws -> VerifiedFile {
        let url = try sourceURL(file.path, in: folder)
        var value = stat()
        // Follow a cache/import symlink, recording the identity of the actual bytes that hashing reads.
        guard stat(url.path, &value) == 0, (value.st_mode & S_IFMT) == S_IFREG else {
            throw CoreMLError.invalidAsset("missing or unreadable model file: \(file.path)")
        }
        guard value.st_size == file.bytes else {
            throw CoreMLError.invalidAsset("size mismatch: \(file.path)")
        }
        return VerifiedFile(expected: file, device: Int64(value.st_dev), inode: UInt64(value.st_ino),
                            size: Int64(value.st_size), modifiedSeconds: Int64(value.st_mtimespec.tv_sec),
                            modifiedNanoseconds: Int64(value.st_mtimespec.tv_nsec),
                            changedSeconds: Int64(value.st_ctimespec.tv_sec),
                            changedNanoseconds: Int64(value.st_ctimespec.tv_nsec))
    }

    private static func encoded<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        return try encoder.encode(value)
    }

    private static func cacheDirectory(_ folder: URL) throws -> URL {
        let cache = folder.appendingPathComponent("Compiled", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        return cache
    }

    private static func withLock<T>(at url: URL, _ body: () throws -> T) throws -> T {
        let descriptor = Darwin.open(url.path, O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw CoreMLError.invalidAsset("cannot open model cache lock") }
        defer { Darwin.close(descriptor) }
        while flock(descriptor, LOCK_EX) != 0 {
            guard errno == EINTR else { throw CoreMLError.invalidAsset("cannot lock model cache") }
        }
        defer { _ = flock(descriptor, LOCK_UN) }
        return try body()
    }
}
