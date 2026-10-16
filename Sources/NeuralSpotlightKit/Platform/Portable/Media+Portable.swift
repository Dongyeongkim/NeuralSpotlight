// Audio and video without AVFoundation (Linux, Windows): ffmpeg decodes, ffprobe measures. They're looked for beside
// the helper, in $NEURALSPOTLIGHT_FFMPEG, then on PATH.
#if !canImport(AVFoundation)
import Foundation

enum FFmpeg {
    static let ffmpeg = find("ffmpeg")
    static let ffprobe = find("ffprobe")

    static func find(_ name: String) -> String? {
        let environment = ProcessInfo.processInfo.environment
        var folders: [String] = []
        if let chosen = environment["NEURALSPOTLIGHT_FFMPEG"] ?? environment["DIGUP_FFMPEG"], !chosen.isEmpty {
            folders.append(chosen)
        }
        if let executable = Bundle.main.executableURL { folders.append(executable.deletingLastPathComponent().path) }
        for folder in folders {
            let candidate = folder + "/" + FileSystem.program(name)
            if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
        }
        return FileSystem.executable(name)
    }

    /// Runs a tool and hands its standard output to `read` as it comes (a long decode never sits in memory whole).
    /// Throws when the tool is missing or fails.
    static func run(_ tool: String?, _ arguments: [String], read: (Data) throws -> Void) throws {
        guard let tool else { throw ExtractError.unreadable("ffmpeg isn't installed (audio and video need it)") }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: tool)
        process.arguments = arguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        try process.run()
        let reader = output.fileHandleForReading
        while true {
            let data = reader.readData(ofLength: 1 << 16)
            if data.isEmpty { break }
            try read(data)
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ExtractError.unreadable("\((tool as NSString).lastPathComponent) failed (\(process.terminationStatus))")
        }
    }

    /// The whole standard output, as text.
    static func output(_ tool: String?, _ arguments: [String]) throws -> String {
        var data = Data()
        try run(tool, arguments) { data.append($0) }
        return String(decoding: data, as: UTF8.self)
    }
}

extension MediaInfo {
    static func duration(of url: URL) -> Double? {
        guard let text = try? FFmpeg.output(FFmpeg.ffprobe, ["-v", "error", "-show_entries", "format=duration", "-of",
                                                             "default=noprint_wrappers=1:nokey=1", url.path])
        else { return nil }
        let seconds = Double(text.trimmingCharacters(in: .whitespacesAndNewlines))
        return seconds.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
    }

    /// Which kinds of streams the file has.
    static func streams(of url: URL) -> (audio: Bool, video: Bool) {
        guard let text = try? FFmpeg.output(FFmpeg.ffprobe, ["-v", "error", "-show_entries", "stream=codec_type", "-of",
                                                             "default=noprint_wrappers=1:nokey=1", url.path])
        else { return (false, false) }
        return (text.contains("audio"), text.contains("video"))
    }
}

extension Estimator {
    static func duration(of path: String) -> Double {
        MediaInfo.duration(of: URL(fileURLWithPath: path)) ?? 0
    }
}

extension AudioExtractor {
    /// Decodes audio (a file, or a video's soundtrack) to 16 kHz mono through ffmpeg and writes `window`-second WAV
    /// pieces `hop` seconds apart, skipping silent ones; one model input ignores anything past 30 s, so windows stay
    /// ≤ 30 s. Streams: memory stays at one window, even for a 3-hour podcast.
    static func windows(of url: URL, window: Double, hop: Double, minimum: Double = 2,
                        range rangeClosure: ((Double) -> Range<Double>)? = nil, to folder: URL) async throws
        -> (pieces: [MediaPiece], duration: Double) {
        guard MediaInfo.streams(of: url).audio else { return ([], 0) }
        let duration = MediaInfo.duration(of: url) ?? 0
        let wanted = rangeClosure?(duration)
        if let wanted, wanted.isEmpty { return ([], duration) }
        let windowSamples = Int(window * sampleRate), hopSamples = Int(hop * sampleRate)
        let baseSample = Int((wanted?.lowerBound ?? 0) * sampleRate)
        var buffer: [Int16] = []
        var offset = baseSample   // sample index of buffer[0] in the file
        var pieces: [MediaPiece] = []
        func emit(_ count: Int) throws {
            let slice = buffer[0..<count]
            guard rms(slice) > 0.003 else { return }   // ≈ −50 dBFS: silence isn't worth a vector
            let out = folder.appendingPathComponent(UUID().uuidString + ".wav")
            try writeWAV(slice, to: out)
            pieces.append(MediaPiece(path: out.path, start: Double(offset) / sampleRate,
                                     end: Double(offset + count) / sampleRate))
        }
        var odd: UInt8?   // a byte left over when a read ends mid-sample
        var args = ["-v", "error", "-nostdin"]
        if let wanted, wanted.lowerBound > 0 { args += ["-ss", String(wanted.lowerBound)] }
        if let wanted, duration > 0, wanted.upperBound < duration { args += ["-to", String(wanted.upperBound)] }
        args += ["-i", url.path, "-vn", "-ac", "1", "-ar", "\(Int(sampleRate))", "-f", "s16le", "-"]
        try FFmpeg.run(FFmpeg.ffmpeg, args) { data in
            var bytes: [UInt8] = []
            bytes.reserveCapacity(data.count + 1)
            if let first = odd { bytes.append(first); odd = nil }
            bytes.append(contentsOf: data)
            if bytes.count % 2 == 1 { odd = bytes.removeLast() }
            buffer.reserveCapacity(buffer.count + bytes.count / 2)
            var i = 0
            while i + 1 < bytes.count {
                buffer.append(Int16(bitPattern: UInt16(bytes[i]) | UInt16(bytes[i + 1]) << 8))
                i += 2
            }
            while buffer.count >= windowSamples {
                try emit(windowSamples)
                buffer.removeFirst(hopSamples)
                offset += hopSamples
            }
        }
        // The tail: new audio beyond the previous window's overlap, or the whole file if it was shorter than a window.
        let alreadyCovered = offset == baseSample ? 0 : windowSamples - hopSamples
        if buffer.count - alreadyCovered >= Int(minimum * sampleRate) { try emit(buffer.count) }
        return (pieces, duration > 0 ? duration : Double(offset + buffer.count) / sampleRate)
    }
}

extension VideoExtractor {
    /// Samples a frame every `interval` seconds through ffmpeg and keeps one per shot: a frame that looks like the
    /// previous kept one (8×8 average hash within 4 bits) only extends it.
    static func keyframes(of url: URL, every interval: Double, maxPixel: Int, maxFrames: Int = 1200,
                          range rangeClosure: ((Double) -> Range<Double>)? = nil, to folder: URL)
        async throws -> (frames: [MediaPiece], duration: Double, hasAudio: Bool) {
        let streams = MediaInfo.streams(of: url)
        let duration = MediaInfo.duration(of: url) ?? 0
        guard duration > 0, streams.video else { return ([], duration, streams.audio) }
        let wanted = rangeClosure?(duration) ?? (0..<duration)
        guard !wanted.isEmpty else { return ([], duration, streams.audio) }
        let span = max(0.001, wanted.upperBound - wanted.lowerBound)
        let step = max(interval, span / Double(maxFrames))
        let first = wanted.lowerBound > 0 ? wanted.lowerBound : min(0.5, duration / 2)
        let endTime = min(duration, wanted.upperBound)
        let prefix = "frame-\(UUID().uuidString)-"
        var args = ["-v", "error", "-nostdin"]
        if wanted.lowerBound > 0 { args += ["-ss", String(wanted.lowerBound)] }
        if endTime < duration { args += ["-to", String(endTime)] }
        let filterStart = wanted.lowerBound > 0 ? 0.0 : first
        args += ["-i", url.path, "-an", "-vf",
                 "fps=fps=\(1 / step):start_time=\(filterStart):round=up,"
                 + "scale=\(maxPixel):\(maxPixel):force_original_aspect_ratio=decrease",
                 "-q:v", "3", "-f", "image2", folder.appendingPathComponent(prefix + "%06d.jpg").path]
        try FFmpeg.run(FFmpeg.ffmpeg, args) { _ in }
        let names = try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasPrefix(prefix) }.sorted()
        var frames: [MediaPiece] = []
        var lastHash: UInt64?
        for (index, name) in names.enumerated() {
            let path = folder.appendingPathComponent(name).path
            let start = first + Double(index) * step
            guard start < endTime else {   // fps can fill a slot past the end
                try? FileManager.default.removeItem(atPath: path)
                continue
            }
            if let image = RGBImage.load(path) {
                let hash = image.averageHash()
                if let lastHash, (hash ^ lastHash).nonzeroBitCount <= 4 {
                    try? FileManager.default.removeItem(atPath: path)
                    continue
                }
                lastHash = hash
            }
            frames.append(MediaPiece(path: path, start: start, end: endTime))
        }
        // Each kept frame stands for the stretch until the next kept frame.
        for index in frames.indices.dropLast() { frames[index].end = frames[index + 1].start }
        return (frames, duration, streams.audio)
    }

    /// The frame shown at `seconds`, its long side at most `pixels`, as a JPEG for a preview.
    static func frame(at seconds: Double, of url: URL, pixels: Int, to out: URL) async throws {
        let at = max(0, seconds)
        let rest = ["-frames:v", "1", "-vf",
                    "scale='min(iw,\(pixels))':'min(ih,\(pixels))':force_original_aspect_ratio=decrease",
                    "-q:v", "3", "-y", out.path]
        // Seeking before reading starts at the keyframe before the moment and decodes on to it: quick at any length. An
        // MPEG-TS has no index, so its seek lands on the moment itself, and nothing decodes until the next keyframe (a
        // later frame is written, or none): seek to a while before and read on. ffmpeg may then exit with an error for
        // the frames it couldn't decode on the way, though the frame it wrote is the right one.
        let lead = Rules.isMPEGTransportStream(url) ? min(at, 12) : 0   // x264 keeps up to 250 frames between keyframes
        try? FFmpeg.run(FFmpeg.ffmpeg, ["-v", "error", "-nostdin", "-ss", String(at - lead), "-i", url.path]
                                       + (lead > 0 ? ["-ss", String(lead)] : []) + rest) { _ in }
        if !FileManager.default.fileExists(atPath: out.path) {
            // Nothing written: read from the start to the moment.
            try FFmpeg.run(FFmpeg.ffmpeg, ["-v", "error", "-nostdin", "-i", url.path, "-ss", String(at)] + rest) { _ in }
        }
        guard FileManager.default.fileExists(atPath: out.path) else {
            throw ExtractError.unreadable("no frame at \(seconds) s")
        }
    }
}
#endif
