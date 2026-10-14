#if canImport(CryptoKit)
import CryptoKit
#endif
import Foundation

/// Shared model-file download, cache and integrity utilities. The native model manifest is `CoreMLModelFiles`.
public enum ModelFiles {

    public struct File: Sendable, Equatable {
        public let name: String
        public let bytes: Int64
        public let sha256: String
        public let repository: String
        public let revision: String
        public let remotePath: String

        public init(name: String, bytes: Int64, sha256: String,
                    repository: String, revision: String,
                    remotePath: String? = nil) {
            self.name = name
            self.bytes = bytes
            self.sha256 = sha256
            self.repository = repository
            self.revision = revision
            self.remotePath = remotePath ?? name
        }

        /// Hugging Face's address for this file at the pinned revision (it redirects to a CDN that takes HTTP ranges).
        public var url: URL {
            URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(remotePath)")!
        }
    }

    /// Where the app keeps them (`AppPaths`).
    public static let installed = AppPaths.applicationSupport.appendingPathComponent("Models")

    /// The pinned revision's folder in the Hugging Face cache (`hf download` puts the files there): `$HF_HUB_CACHE`,
    /// `$HF_HOME/hub`, or `~/.cache/huggingface/hub`.
    public static func huggingFaceSnapshot(repository: String, revision: String) -> URL {
        let environment = ProcessInfo.processInfo.environment
        let hub = environment["HF_HUB_CACHE"].map { URL(fileURLWithPath: $0) }
            ?? environment["HF_HOME"].map { URL(fileURLWithPath: $0).appendingPathComponent("hub") }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub")
        return hub.appendingPathComponent("models--\(repository.replacingOccurrences(of: "/", with: "--"))/snapshots/\(revision)")
    }

    /// `file` in the Hugging Face cache, when it's there at the right size (its checksum is for the caller to check).
    public static func cachedCopy(of file: File) -> URL? {
        let url = huggingFaceSnapshot(repository: file.repository, revision: file.revision)
            .appendingPathComponent(file.remotePath).resolvingSymlinksInPath()
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value
        return size == file.bytes ? url : nil
    }

    /// The SHA-256 of a file as lowercase hex, read in 8 MB pieces (a 555 MB file takes about half a second).
    public static func sha256(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let done = try autoreleasepool { () -> Bool in
                guard let data = try handle.read(upToCount: 8 << 20), !data.isEmpty else { return true }
                hasher.update(data: data)
                return false
            }
            if done { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
