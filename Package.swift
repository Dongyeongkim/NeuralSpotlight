// swift-tools-version: 6.2
import PackageDescription

// NeuralSpotlight builds on macOS (the app, the CLI and its helpers) and, since the port, the engine and the CLI on
// Linux and Windows too. The manifest picks each platform's pieces: Core ML on macOS, the C libraries the engine
// needs where Apple's frameworks aren't there (SQLite, stb, miniz, PDFium: scripts/fetch-vendor.sh puts them in
// Vendor/), and the app, which is macOS only.
let vendor = "\(Context.packageDirectory)/Vendor"

#if os(macOS)
let engineDependencies: [Target.Dependency] = []
let engineLink: [LinkerSetting] = []
let portableTargets: [Target] = []
#else
// Portable builds keep extraction, storage, search and the external embedding-worker protocol.
// Core ML is unavailable there; no native embedding runtime is bundled.
#if os(Windows)
// No rpath on Windows: the PDFium DLL goes beside neuralspotlight.exe, or on PATH in dev runs.
let runpath: [String] = []
#else
let runpath = ["-Xlinker", "-rpath", "-Xlinker", "$ORIGIN",
               "-Xlinker", "-rpath", "-Xlinker", "\(vendor)/pdfium/lib"]
#endif
let engineDependencies: [Target.Dependency] = ["CSQLite", "CStb", "CMiniz", "CPDFium"]
let engineLink: [LinkerSetting] = [.unsafeFlags(["-L\(vendor)/pdfium/lib"] + runpath)]
let portableTargets: [Target] = [
    // SQLite with FTS5 (Windows has no system SQLite, Linux's may lack FTS5 or its headers).
    .target(name: "CSQLite", path: "Vendor/sqlite", cSettings: [
        .define("SQLITE_ENABLE_FTS5"), .define("SQLITE_THREADSAFE", to: "1"), .define("SQLITE_DQS", to: "0"),
        .define("SQLITE_OMIT_LOAD_EXTENSION"), .unsafeFlags(["-w"]),
    ]),
    // Image decoding, resizing and PNG/JPEG writing (stb_image, stb_image_resize2, stb_image_write). On Windows its file
    // names are UTF-8, opened through the wide-character calls (else a Bengali or Arabic name won't open).
    .target(name: "CStb", path: "Vendor/stb", cSettings: [
        .define("STBI_WINDOWS_UTF8", .when(platforms: [.windows])),
        .define("STBIW_WINDOWS_UTF8", .when(platforms: [.windows])), .unsafeFlags(["-w"]),
    ]),
    // Zip reading, for docx and odt.
    .target(name: "CMiniz", path: "Vendor/miniz", cSettings: [.unsafeFlags(["-w"])]),
    // PDF text and page rendering (Chromium's PDFium, prebuilt).
    .systemLibrary(name: "CPDFium", path: "Sources/CPDFium"),
]
#endif

#if os(macOS)
let runtimeDependencies: [Target.Dependency] = ["CoreMLRuntime"]
let appProducts: [Product] = [.executable(name: "NeuralSpotlightApp", targets: ["NeuralSpotlightApp"])]
let appDependencies: [Package.Dependency] = [
    // In-app updates (the app only). build.sh embeds the framework in Contents/Frameworks and signs it.
    .package(url: "https://github.com/sparkle-project/Sparkle", exact: "2.10.0"),
]
let appTargets: [Target] = [
    .target(name: "CoreMLRuntime", dependencies: ["NeuralSpotlightKit"], exclude: ["FluidUse-LICENSE.txt"]),
    .testTarget(name: "CoreMLRuntimeTests", dependencies: ["CoreMLRuntime"], exclude: ["Reference"]),
    // The menubar + hotkey app. build.sh wraps it into build.noindex/NeuralSpotlight.app. (The product can't be called
    // "NeuralSpotlight": on a case-insensitive disk it would collide with the `neuralspotlight` CLI binary.)
    .executableTarget(
        name: "NeuralSpotlightApp",
        dependencies: ["NeuralSpotlightKit", .product(name: "Sparkle", package: "Sparkle")],
        swiftSettings: [.defaultIsolation(MainActor.self)],
        linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/../Frameworks"])]
    ),
]
#else
let runtimeDependencies: [Target.Dependency] = []
let appProducts: [Product] = []
let appDependencies: [Package.Dependency] = []
let appTargets: [Target] = []
#endif

let package = Package(
    name: "NeuralSpotlight",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "neuralspotlight", targets: ["neuralspotlight"]),
        .library(name: "NeuralSpotlightKit", targets: ["NeuralSpotlightKit"]),
    ] + appProducts,
    dependencies: appDependencies + [
        // 2.25+ introduces a header-only NumKong target that Xcode's SwiftPM linker cannot consume.
        .package(url: "https://github.com/unum-cloud/USearch", exact: "2.24.0"),
    ],
    targets: portableTargets + [
        // The engine: crawler, extractors, index store, the embedder interface and worker client, search. No UI.
        // Platform/ holds what differs by OS (Apple's frameworks on macOS, the C libraries above elsewhere).
        .target(name: "NeuralSpotlightKit", dependencies: engineDependencies + [.product(name: "USearch", package: "USearch")],
                linkerSettings: engineLink),
        // Developer CLI: estimate / index / search / status / eval. Doubles as the eval harness, and the app runs it as
        // its indexing helper and its query encoder.
        .executableTarget(name: "neuralspotlight", dependencies: ["NeuralSpotlightKit"] + runtimeDependencies),
        // Pictures/ holds ImageTests' pictures (scripts/testbed/make_test_pictures.swift) and Videos/ VideoTests' video,
        // read by path.
        .testTarget(name: "NeuralSpotlightKitTests", dependencies: ["NeuralSpotlightKit"], exclude: ["Pictures", "Videos"]),
    ] + appTargets
)
