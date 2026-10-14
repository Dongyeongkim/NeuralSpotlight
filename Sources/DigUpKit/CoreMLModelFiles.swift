import Foundation

/// Fixed-shape, overflow-safe FP16 text model. The lookup table is BF16 on disk; Core ML receives FP16 inputs.
/// Download paths are relative to the shared Models directory; `locate` returns its CoreML subdirectory.
public enum CoreMLModelFiles {
    public static let repository = "FluidInference/embeddinggemma-2-coreml"
    public static let revision = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"
    public static let package = "EmbeddingGemma2Text.mlpackage"
    public static let visionPackage = "EmbeddingGemma2Vision.mlpackage"
    public static let audioPackage = "EmbeddingGemma2Audio.mlpackage"
    public static let installed = ModelFiles.installed.appendingPathComponent("CoreML")

    private static func file(_ path: String, _ bytes: Int64, _ sha256: String) -> ModelFiles.File {
        ModelFiles.File(name: "CoreML/\(path)", bytes: bytes, sha256: sha256,
                        repository: repository, revision: revision, remotePath: path)
    }

    public static let text: [ModelFiles.File] = [
        file("tokenizer.json", 32_170_510, "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
        file("embeddings.bf16", 268_435_456, "56ebbdbfd706c827f3fae1d97aa39135c7ddc0f1e6d9b12f8565f84bbfbda877"),
        file("\(package)/Manifest.json", 617, "83eace80b69b46f8415af620ba1b40d7c490d00f7c00ba9dedd2e29da6e9ac19"),
        file("\(package)/Data/com.apple.CoreML/model.mlmodel", 7_323_423,
             "0df9aca02807cf490db6d4f1c68607a8c7f486ad23b376d1968fd3be5eb14677"),
        file("\(package)/Data/com.apple.CoreML/weights/weight.bin", 276_815_424,
             "75b97ab036ca87eae60560c4034a63d604201498dc4b6f6971dd0b1e4558d61f"),
    ]
    public static let vision: [ModelFiles.File] = [
        file("position_embeddings.f16", 3_145_728,
             "17de9480acc23638c0634a97ed89e51ed664433256d7d6f109587dcd91c79124"),
        file("\(visionPackage)/Manifest.json", 617,
             "376afe7b51608637312cd15a0a89b1d7863f9779502aaf5f12fa6b14c4272d8d"),
        file("\(visionPackage)/Data/com.apple.CoreML/model.mlmodel", 2_053_174,
             "0085b5e04a13d000fe5969adeaa13d897f5a3cb7be60c76c8685c023747f89ef"),
        file("\(visionPackage)/Data/com.apple.CoreML/weights/weight.bin", 304_078_400,
             "5b5b6dd8d80aa438bf91ba143fbe90d56a2e25bfcf685a56b990aad71cf75232"),
    ]
    public static let audio: [ModelFiles.File] = [
        file("\(audioPackage)/Manifest.json", 617,
             "65c8e9811516e7916a3d742261c7d5a02d4e43c113b437330ce86f98b3da3514"),
        file("\(audioPackage)/Data/com.apple.CoreML/model.mlmodel", 1_059_089,
             "d40896e5c88d62b02d15cc975f27e254c28a5774a1702e09bf5483071623548d"),
        file("\(audioPackage)/Data/com.apple.CoreML/weights/weight.bin", 586_817_408,
             "41d7b552caabe376e377dcbe1c8c5f9879853bc4f384a161bb1f886f19e84b56"),
    ]
    public static let all = text + vision + audio
    public static let totalBytes = all.reduce(Int64(0)) { $0 + $1.bytes }

    public static var supported: Bool {
        #if os(macOS) && arch(arm64)
        if #available(macOS 15, *) { return true }
        #endif
        return false
    }

    /// The directory itself contains tokenizer.json, embeddings.bf16 and the ML package.
    public static var folder: URL {
        let environment = ProcessInfo.processInfo.environment
        if let path = environment["DIGUP_COREML_MODELS"], !path.isEmpty { return URL(fileURLWithPath: path) }
        if let path = environment["DIGUP_MODELS"], !path.isEmpty {
            return URL(fileURLWithPath: path).appendingPathComponent("CoreML")
        }
        return installed
    }

    /// Size checks for readiness; the runtime verifies hashes before loading the model.
    public static func complete(in folder: URL, textOnly: Bool = false) -> Bool {
        (textOnly ? text : all).allSatisfy { file in
            let url = folder.appendingPathComponent(file.remotePath)
            return (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value == file.bytes
        }
    }

    public static var complete: Bool { supported && complete(in: folder) }
    public static var textComplete: Bool { supported && complete(in: folder, textOnly: true) }

    /// Missing files relative to the shared Models root, used by the app's download manager.
    public static func missing(in modelsRoot: URL) -> [ModelFiles.File] {
        all.filter { file in
            let url = modelsRoot.appendingPathComponent(file.name)
            return (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.int64Value != file.bytes
        }
    }

    public static func locate(textOnly: Bool = false) throws -> URL {
        guard supported else { throw WorkerError.notFound("Core ML embedding needs Apple silicon and macOS 15 or newer") }
        guard complete(in: folder, textOnly: textOnly) else {
            throw WorkerError.notFound("Core ML \(textOnly ? "text" : "multimodal") model is incomplete in \(folder.path)")
        }
        return folder
    }
}
