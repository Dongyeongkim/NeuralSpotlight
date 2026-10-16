import CoreServices
import Foundation

/// File-level FSEvents on the chosen folders. Reports which paths changed, a batch at a time; the engine debounces
/// and syncs. Costs nothing while nothing changes (no polling), and hidden files (.DS_Store, editors' temp files)
/// are ignored, since browsing a folder in Finder writes those.
nonisolated final class FolderWatcher: @unchecked Sendable {
    struct Event: Sendable {
        let path: String
        let flags: FSEventStreamEventFlags
        let id: FSEventStreamEventId

        var requiresRescan: Bool {
            let mask = FSEventStreamEventFlags(
                kFSEventStreamEventFlagMustScanSubDirs
                | kFSEventStreamEventFlagUserDropped
                | kFSEventStreamEventFlagKernelDropped
            )
            return (flags & mask) != 0
        }
    }

    private var stream: FSEventStreamRef?
    private let handler: @Sendable ([Event]) -> Void
    private(set) var lastEventId: FSEventStreamEventId

    convenience init?(paths: [String], latency: TimeInterval, queue: DispatchQueue,
                      handler: @escaping @Sendable ([String]) -> Void) {
        self.init(paths: paths, latency: latency, queue: queue, onEvents: { events in
            handler(events.map(\.path))
        })
    }

    init?(paths: [String], sinceWhen: FSEventStreamEventId = FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
          latency: TimeInterval, queue: DispatchQueue,
          onEvents: @escaping @Sendable ([Event]) -> Void) {
        self.handler = onEvents
        self.lastEventId = sinceWhen
        guard !paths.isEmpty else { return nil }
        var context = FSEventStreamContext(version: 0, info: Unmanaged.passUnretained(self).toOpaque(),
                                           retain: nil, release: nil, copyDescription: nil)
        let flags = kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagFileEvents
            | kFSEventStreamCreateFlagNoDefer
        guard let stream = FSEventStreamCreate(kCFAllocatorDefault, folderWatcherCallback, &context,
                                               paths as CFArray, sinceWhen,
                                               latency, FSEventStreamCreateFlags(flags)) else { return nil }
        self.stream = stream
        FSEventStreamSetDispatchQueue(stream, queue)
        guard FSEventStreamStart(stream) else {
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
            self.stream = nil
            return nil
        }
    }

    deinit {
        guard let stream else { return }
        FSEventStreamStop(stream)
        FSEventStreamInvalidate(stream)
        FSEventStreamRelease(stream)
    }

    fileprivate func received(_ events: [Event]) {
        if let maxId = events.map(\.id).max() { lastEventId = maxId }
        let relevant = events.filter { event in
            event.requiresRescan || !event.path.split(separator: "/").contains { $0.hasPrefix(".") }
        }
        if !relevant.isEmpty { handler(relevant) }
    }
}

private nonisolated func folderWatcherCallback(
    _ stream: ConstFSEventStreamRef, _ info: UnsafeMutableRawPointer?, _ count: Int,
    _ paths: UnsafeMutableRawPointer, _ flags: UnsafePointer<FSEventStreamEventFlags>,
    _ ids: UnsafePointer<FSEventStreamEventId>
) {
    guard let info else { return }
    let watcher = Unmanaged<FolderWatcher>.fromOpaque(info).takeUnretainedValue()
    let list = Unmanaged<CFArray>.fromOpaque(paths).takeUnretainedValue() as? [String] ?? []
    var events: [FolderWatcher.Event] = []
    events.reserveCapacity(min(count, list.count))
    for i in 0..<min(count, list.count) {
        events.append(FolderWatcher.Event(path: list[i], flags: flags[i], id: ids[i]))
    }
    watcher.received(events)
}
