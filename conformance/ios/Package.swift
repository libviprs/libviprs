// swift-tools-version:5.9
import PackageDescription

// A bare executable, wrapped into an .app by run.sh. No Xcode project, so there is nothing to keep in step.
let package = Package(
    name: "Conformance",
    platforms: [.iOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/maplibre/maplibre-gl-native-distribution", exact: "7.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "Conformance",
            dependencies: [.product(name: "MapLibre", package: "maplibre-gl-native-distribution")],
            // run.sh puts MapLibre.framework in the app's Frameworks folder
            linkerSettings: [.unsafeFlags(["-Xlinker", "-rpath", "-Xlinker", "@executable_path/Frameworks"])]
        ),
    ]
)
