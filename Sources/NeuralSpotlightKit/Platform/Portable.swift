// What every platform shares of the platform layer: the shapes the engine asks for, and where DigUp keeps its files.
// Platform/Apple and Platform/Portable implement the same names.
import Foundation

/// A file's own facts, as the crawl records them (`FileSystem.info`).
struct FileInfo {
    let isRegular: Bool   // not a link, device or socket
    let size: Int64
    let mtime: Double     // seconds since 1970, with the fraction the file system keeps
    let inode: UInt64
    let device: UInt64
    /// A cloud placeholder whose contents would be downloaded on read (never read).
    let dataless: Bool
}

#if !canImport(Darwin)
/// Darwin's `autoreleasepool`: elsewhere there's no Objective-C runtime and nothing to drain, so the body just runs.
@discardableResult
public func autoreleasepool<Result>(invoking body: () throws -> Result) rethrows -> Result {
    try body()
}
#endif

extension Data {
    /// Writes the file whole, replacing what was there, so a reader sees the old contents or the new, never half. On
    /// Windows the replace is refused while another program has the file open (error 32, a sharing violation, which
    /// Foundation reports as "You don't have permission"): a virus scanner or the search indexer reading the file just
    /// written. costs.json lost a write that way about once in five test runs. So it's tried again for a second, then
    /// the file is written in place.
    public func writeReplacing(_ url: URL) throws {
        #if os(Windows)
        for _ in 0..<10 {
            if (try? write(to: url, options: .atomic)) != nil { return }
            Thread.sleep(forTimeInterval: 0.1)
        }
        try write(to: url)
        #else
        try write(to: url, options: .atomic)
        #endif
    }
}

/// Where NeuralSpotlight keeps its own files on each OS: the model and the index.
public enum AppPaths {
    /// macOS: ~/Library/Application Support/NeuralSpotlight. Windows: %LOCALAPPDATA%\NeuralSpotlight.
    /// Linux: $XDG_DATA_HOME/neuralspotlight, or ~/.local/share/neuralspotlight.
    public static var applicationSupport: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let environment = ProcessInfo.processInfo.environment
        #if os(macOS)
        let legacy = home.appendingPathComponent("Library/Application Support/DigUp")
        // Reuse the existing index and models in place: a rename must not create a second corpus or download.
        if FileManager.default.fileExists(atPath: legacy.path) { return legacy }
        return home.appendingPathComponent("Library/Application Support/NeuralSpotlight")
        #elseif os(Windows)
        let base = (environment["LOCALAPPDATA"].flatMap { !$0.isEmpty ? URL(fileURLWithPath: $0) : nil })
            ?? home.appendingPathComponent("AppData/Local")
        let legacy = base.appendingPathComponent("DigUp")
        if FileManager.default.fileExists(atPath: legacy.path) { return legacy }
        return base.appendingPathComponent("NeuralSpotlight")
        #else
        let base = (environment["XDG_DATA_HOME"].flatMap { !$0.isEmpty ? URL(fileURLWithPath: $0) : nil })
            ?? home.appendingPathComponent(".local/share")
        let legacy = base.appendingPathComponent("digup")
        if FileManager.default.fileExists(atPath: legacy.path) { return legacy }
        return base.appendingPathComponent("neuralspotlight")
        #endif
    }

    /// The default index folder (".noindex" keeps Spotlight out of it on a Mac).
    public static var defaultIndex: URL {
        #if os(macOS)
        return applicationSupport.appendingPathComponent("index.noindex")
        #else
        return applicationSupport.appendingPathComponent("index")
        #endif
    }
}
