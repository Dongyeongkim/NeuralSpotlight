import Foundation
import USearchC
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif os(Windows)
import WinSDK
#endif
#if canImport(CryptoKit)
import CryptoKit
#endif

/// A disposable HNSW acceleration layer. SQLite remains the source of truth, including vector bytes.
/// Every reload reconciles segment IDs AND vector hashes, so deletion, ID reuse, and changed encoders cannot
/// silently return a stale vector. This type, like Database and Searcher, is confined to one search queue.
final class ANNIndex {
    static let format = "usearch-2.24.0-cos-f16-d768-m32-add256-search2048-v1"
    struct Entry: Codable, Equatable {
        let id: Int64
        let modality: String
        let digest: String
    }
    struct GraphFile: Codable {
        let modality: String
        let name: String
        let digest: String
        let count: Int
    }
    struct Manifest: Codable {
        let format: String
        let source: String?
        let entries: [Entry]
        let graphs: [GraphFile]
    }
    private(set) var graphs: [Modality: USearchGraph] = [:]
    private var entries: [Int64: Entry] = [:]
    private var source: String?
    private var lastSave = Date.distantPast
    private(set) var loadedCache = false
    private(set) var changedVectors = 0
    private let directory: URL

    init(directory: URL) { self.directory = directory }

    func restore() throws {
        try withCacheLock { try restoreLocked() }
    }

    private func restoreLocked() throws {
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: directory.appendingPathComponent("current.json")))
        guard manifest.format == Self.format,
              Set(manifest.entries.map(\.id)).count == manifest.entries.count,
              Set(manifest.graphs.map(\.modality)).count == manifest.graphs.count else { throw ANNFailure.invalidCache }
        var restored: [Modality: USearchGraph] = [:]
        for file in manifest.graphs {
            guard let modality = Modality(rawValue: file.modality), file.name == URL(fileURLWithPath: file.name).lastPathComponent,
                  file.name.hasSuffix(".usearch"), file.count > 0 else { throw ANNFailure.invalidCache }
            let url = directory.appendingPathComponent(file.name)
            guard try ModelFiles.sha256(of: url) == file.digest else { throw ANNFailure.invalidCache }
            let graph = try USearchGraph()
            try graph.load(url)
            guard try graph.count == file.count, try graph.dimensions == Vectors.dimension,
                  manifest.entries.filter({ $0.modality == file.modality }).count == file.count else { throw ANNFailure.invalidCache }
            restored[modality] = graph
        }
        guard manifest.entries.allSatisfy({ entry in
            guard let modality = Modality(rawValue: entry.modality) else { return false }
            return restored[modality] != nil
        }) else { throw ANNFailure.invalidCache }
        graphs = restored
        entries = Dictionary(uniqueKeysWithValues: manifest.entries.map { ($0.id, $0) })
        source = manifest.source
        loadedCache = true
    }

    func synchronize(matrix: VectorStorage, segments: [Int64], modalities: [Modality], source: String?) throws {
        if self.source != source { graphs = [:]; entries = [:]; loadedCache = false }
        self.source = source
        var current: [Int64: Entry] = [:]
        for row in segments.indices {
            let data = Data(bytes: matrix.halves + row * Vectors.dimension,
                            count: Vectors.dimension * MemoryLayout<Float16>.stride)
            let digest = Data(SHA256.hash(data: data)).base64EncodedString()
            current[segments[row]] = Entry(id: segments[row], modality: modalities[row].rawValue, digest: digest)
        }
        changedVectors = 0
        let removals = entries.filter { current[$0.key] != $0.value }.count
        if removals > max(128, entries.count / 5) { graphs = [:]; entries = [:] }
        for (id, old) in entries where current[id] != old {
            if let modality = Modality(rawValue: old.modality) { try graphs[modality]?.remove(id) }
            changedVectors += 1
        }
        let counts = Dictionary(grouping: modalities, by: { $0 }).mapValues(\.count)
        for (modality, count) in counts {
            if graphs[modality] == nil { graphs[modality] = try USearchGraph() }
            // Deleted nodes can remain in the graph until rebuilt; reserve also covers newly assigned keys.
            let additions = current.values.filter { $0.modality == modality.rawValue && entries[$0.id] != $0 }.count
            if let graph = graphs[modality] { try graph.reserve(max(count, graph.count + additions)) }
        }
        for row in segments.indices where current[segments[row]] != entries[segments[row]] {
            try graphs[modalities[row]]?.add(segments[row], vector: matrix.halves + row * Vectors.dimension)
            changedVectors += 1
        }
        for modality in Array(graphs.keys) where counts[modality] == nil { graphs[modality] = nil }
        entries = current
        // Writes are coalesced during indexing. An older cache is safe: the next process replays the full DB diff.
        if changedVectors > 0 && Date().timeIntervalSince(lastSave) >= 30 {
            try? persist()
        }
    }

    func persist() throws {
        try withCacheLock { try persistLocked() }
    }

    private func persistLocked() throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let generation = UUID().uuidString
        var files: [GraphFile] = []
        do {
            for (modality, graph) in graphs {
                let name = "\(generation)-\(modality.rawValue).usearch"
                let url = directory.appendingPathComponent(name)
                try graph.save(url)
                files.append(GraphFile(modality: modality.rawValue, name: name, digest: try ModelFiles.sha256(of: url),
                                       count: try graph.count))
            }
            let manifest = Manifest(format: Self.format, source: source, entries: entries.values.sorted { $0.id < $1.id }, graphs: files)
            try JSONEncoder().encode(manifest).write(to: directory.appendingPathComponent("current.json"), options: .atomic)
            lastSave = Date()
            // Readers finish loading into owned memory under the same lock before old snapshots are removed.
            let keep = Set(files.map(\.name))
            for url in (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
                where url.pathExtension == "usearch" && !keep.contains(url.lastPathComponent) {
                try? FileManager.default.removeItem(at: url)
            }
        } catch {
            for file in files { try? FileManager.default.removeItem(at: directory.appendingPathComponent(file.name)) }
            throw error
        }
    }

    /// The app and CLI may publish caches concurrently. Never wait behind another cache writer on the search
    /// queue: a busy cache is disposable and can be rebuilt, while the canonical SQLite snapshot stays available.
    private func withCacheLock<T>(_ body: () throws -> T) throws -> T {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("cache.lock").path
        #if os(Windows)
        let handle = (path + "\0").withCString(encodedAs: UTF16.self) {
            CreateFileW($0, DWORD(GENERIC_READ | GENERIC_WRITE), 0, nil, DWORD(OPEN_ALWAYS), DWORD(FILE_ATTRIBUTE_NORMAL), nil)
        }
        guard let handle, handle != INVALID_HANDLE_VALUE else { throw ANNFailure.invalidCache }
        defer { CloseHandle(handle) }
        #else
        let descriptor = open(path, O_CREAT | O_RDWR, 0o600)
        guard descriptor >= 0 else { throw ANNFailure.invalidCache }
        defer { _ = flock(descriptor, LOCK_UN); _ = close(descriptor) }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else { throw ANNFailure.invalidCache }
        #endif
        return try body()
    }

    func vector(segment: Int64, modality: Modality, into buffer: UnsafeMutablePointer<Float>) throws {
        guard let graph = graphs[modality] else { throw ANNFailure.invalidCache }
        try graph.vector(segment, into: buffer)
    }
}

enum ANNFailure: Error { case invalidCache, native(String) }

/// Own the native handle explicitly; the upstream Swift facade does not currently release it in deinit.
final class USearchGraph {
    private let handle: UnsafeMutableRawPointer
    init() throws {
        var options = usearch_init_options_t(metric_kind: usearch_metric_cos_k, metric: nil,
            quantization: usearch_scalar_f16_k, dimensions: Vectors.dimension, connectivity: 32,
            expansion_add: 256, expansion_search: 2048, multi: false)
        var error: UnsafePointer<CChar>?
        let pointer = usearch_init(&options, &error)
        if let error { throw ANNFailure.native(String(cString: error)) }
        guard let pointer else { throw ANNFailure.invalidCache }
        handle = pointer
    }
    deinit { var error: UnsafePointer<CChar>?; usearch_free(handle, &error) }
    private func checked<T>(_ body: (UnsafeMutablePointer<UnsafePointer<CChar>?>) -> T) throws -> T {
        var error: UnsafePointer<CChar>?
        let result = body(&error)
        if let error { throw ANNFailure.native(String(cString: error)) }
        return result
    }
    var count: Int { get throws { try checked { usearch_size(handle, $0) } } }
    var dimensions: Int { get throws { try checked { usearch_dimensions(handle, $0) } } }
    func reserve(_ count: Int) throws { try checked { usearch_reserve(handle, count, $0) } }
    func add(_ key: Int64, vector: UnsafePointer<Float16>) throws {
        try checked { usearch_add(handle, UInt64(bitPattern: key), vector, usearch_scalar_f16_k, $0) }
    }
    func remove(_ key: Int64) throws { _ = try checked { usearch_remove(handle, UInt64(bitPattern: key), $0) } }
    func save(_ url: URL) throws { try checked { usearch_save(handle, url.path, $0) } }
    func load(_ url: URL) throws { try checked { usearch_load(handle, url.path, $0) } }
    func vector(_ key: Int64, into buffer: UnsafeMutablePointer<Float>) throws {
        let found = try checked { usearch_get(handle, UInt64(bitPattern: key), 1, buffer, usearch_scalar_f32_k, $0) }
        guard found == 1 else { throw ANNFailure.invalidCache }
    }
    private final class Filter {
        let includes: (Int64) -> Bool
        init(_ includes: @escaping (Int64) -> Bool) { self.includes = includes }
    }
    func search(_ query: [Float], count: Int, includes: ((Int64) -> Bool)? = nil) throws -> [Int64] {
        guard count > 0, query.count == Vectors.dimension else { return [] }
        var keys = [UInt64](repeating: 0, count: count)
        var distances = [Float](repeating: 0, count: count)
        guard let includes else {
            let found = try checked { usearch_search(handle, query, usearch_scalar_f32_k, count, &keys, &distances, $0) }
            return keys.prefix(found).map { Int64(bitPattern: $0) }
        }
        let filter = Filter(includes)
        let found = try checked { error in
            usearch_filtered_search(handle, query, usearch_scalar_f32_k, count, { key, state in
                guard let state else { return 0 }
                return Unmanaged<Filter>.fromOpaque(state).takeUnretainedValue().includes(Int64(bitPattern: key)) ? 1 : 0
            }, Unmanaged.passUnretained(filter).toOpaque(), &keys, &distances, error)
        }
        return keys.prefix(found).map { Int64(bitPattern: $0) }
    }
}
