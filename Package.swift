// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "Swift-TTS",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(
            name: "mlx-swift-audio",
            targets: ["Swift-TTS","ESpeakNG"]),

    ],
    dependencies: [
         .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.2"),
         .package(url: "https://github.com/ml-explore/mlx-swift-examples.git", branch: "main"),
         .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.22")),
    ],
    targets: [
        .binaryTarget(
            name: "ESpeakNG",
            path: "mlx_audio_swift/tts/Swift-TTS/Kokoro/Frameworks/ESpeakNG.xcframework"
        ),
        .target(
            name: "Swift-TTS",
            dependencies: [
      .product(name: "MLX", package: "mlx-swift"),
                    .product(name: "MLXNN", package: "mlx-swift"),
                    .product(name: "MLXRandom", package: "mlx-swift"),
                    .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                    .product(name: "MLXLLM", package: "mlx-swift-examples"),
                    .product(name: "Transformers", package: "swift-transformers"),
                "ESpeakNG"
            ],
            path: "mlx_audio_swift/tts/Swift-TTS",
            exclude: ["Preview Content", "Assets.xcassets", "Swift_TTSApp.swift", "Swift_TTS.entitlements"],
            resources: [
                .process("Kokoro/Resources"),                  // Kokoro voices
            ]
        ),
        .testTarget(
            name: "Swift-TTS-Tests",
            dependencies: ["Swift-TTS"],
            path: "mlx_audio_swift/tts/Tests"
        ),

    ]
)
