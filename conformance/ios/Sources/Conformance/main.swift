// Issue #1037, iOS half: the golden PMTiles, drawn by MapLibre Native iOS with the shared black on
// white style. The server is web/vector-viewer/e2e/server.mjs on the host's loopback, which the
// simulator shares. Same three checks, same thresholds and same camera as the Android test.
import CryptoKit
import MapLibre
import UIKit

let store = "http://127.0.0.1:4174"
let glyphs = "http://127.0.0.1:4173/fonts/{fontstack}/{range}.pbf"
let evidenceDir = URL(fileURLWithPath: ProcessInfo.processInfo.environment["EVIDENCE_DIR"] ?? NSTemporaryDirectory())

struct Failure: Error { let message: String }

func check(_ ok: Bool, _ message: @autoclosure () -> String) throws {
    if !ok { throw Failure(message: message()) }
}

func fetch(_ path: String) async throws -> (Data, HTTPURLResponse) {
    var req = URLRequest(url: URL(string: store + path)!)
    req.cachePolicy = .reloadIgnoringLocalCacheData
    let (data, resp) = try await URLSession.shared.data(for: req)
    return (data, resp as! HTTPURLResponse)
}

/// The shared style with its two placeholders filled, as the Rust helper does.
func styleFor(_ archiveUrl: String) throws -> String {
    let path = Bundle.main.path(forResource: "black_on_white.style", ofType: "json")!
    var style = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as! [String: Any]
    var sources = style["sources"] as! [String: Any]
    var drawing = sources["drawing"] as! [String: Any]
    drawing["url"] = "pmtiles://" + archiveUrl
    sources["drawing"] = drawing
    style["sources"] = sources
    style["glyphs"] = glyphs
    return String(data: try JSONSerialization.data(withJSONObject: style), encoding: .utf8)!
}

/// Requests the store saw for the golden archive since the last reset.
func goldenLog() async throws -> [[String: Any]] {
    let (data, _) = try await fetch("/__log")
    let all = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
    return all.filter { ($0["path"] as? String)?.hasSuffix("/golden.pmtiles") == true }
}

struct Shot {
    var lineFeatures = 0, textFeatures = 0, black = 0, white = 0, other = 0, coloured = 0, total = 0
    var width = 0, height = 0
    var png = Data()
}

final class Loader: NSObject, MLNMapViewDelegate {
    var styled: CheckedContinuation<Void, Never>?
    func mapView(_ mapView: MLNMapView, didFinishLoading style: MLNStyle) { styled?.resume(); styled = nil }
}

@MainActor
func visibleFeatures(_ map: MLNMapView, _ layer: String) -> Int {
    map.visibleFeatures(in: map.bounds, styleLayerIdentifiers: [layer]).count
}

/// Loads the style in a fresh map at a fixed camera, waits for drawing, and measures it.
@MainActor
func render(window: UIWindow, archiveUrl: String, expectContent: Bool) async throws -> Shot {
    let json = try styleFor(archiveUrl)
    let loader = Loader()
    let map = MLNMapView(frame: window.bounds, styleJSON: json)
    map.delegate = loader
    map.logoView.isHidden = true
    map.attributionButton.isHidden = true
    map.compassView.isHidden = true
    // Known camera: the middle of the drawing, zoom 1.5. MapLibre Native iOS zoomLevel is the same web mercator scale as Android's.
    map.setCenter(CLLocationCoordinate2D(latitude: 0, longitude: 0), zoomLevel: 1.5, animated: false)
    window.rootViewController!.view.addSubview(map)
    await withCheckedContinuation { c in
        loader.styled = c
        Task { try? await Task.sleep(nanoseconds: 60_000_000_000); loader.styled?.resume(); loader.styled = nil }
    }
    var shot = Shot()
    let deadline = Date().addingTimeInterval(expectContent ? 90 : 12)
    while Date() < deadline {
        try await Task.sleep(nanoseconds: 500_000_000)
        shot.lineFeatures = visibleFeatures(map, "linework")
        shot.textFeatures = visibleFeatures(map, "annotations")
        if expectContent && shot.lineFeatures > 0 && shot.textFeatures > 0 {
            try await Task.sleep(nanoseconds: 3_000_000_000) // let the rest of the tiles and the glyphs land
            break
        }
    }
    shot.lineFeatures = visibleFeatures(map, "linework")
    shot.textFeatures = visibleFeatures(map, "annotations")
    let fmt = UIGraphicsImageRendererFormat.default()
    fmt.opaque = true
    let image = UIGraphicsImageRenderer(bounds: map.bounds, format: fmt).image { _ in
        map.drawHierarchy(in: map.bounds, afterScreenUpdates: true)
    }
    map.removeFromSuperview()
    shot.png = image.pngData()!
    measure(image.cgImage!, &shot)
    return shot
}

func measure(_ cg: CGImage, _ s: inout Shot) {
    let w = cg.width, h = cg.height
    var px = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    s.width = w; s.height = h; s.total = w * h
    for i in 0..<(w * h) {
        let r = Int(px[i * 4]), g = Int(px[i * 4 + 1]), b = Int(px[i * 4 + 2])
        let mx = max(r, g, b), mn = min(r, g, b)
        if mx - mn > 24 { s.coloured += 1 }
        else if mx <= 64 { s.black += 1 }
        else if mn >= 245 { s.white += 1 }
        else { s.other += 1 }
    }
}

func save(_ name: String, _ s: Shot, _ extra: [String: Any] = [:]) throws {
    try FileManager.default.createDirectory(at: evidenceDir, withIntermediateDirectories: true)
    try s.png.write(to: evidenceDir.appendingPathComponent(name + ".png"))
    var j = extra
    j["lineFeatures"] = s.lineFeatures; j["textFeatures"] = s.textFeatures
    j["blackPx"] = s.black; j["whitePx"] = s.white; j["greyPx"] = s.other; j["colouredPx"] = s.coloured
    j["totalPx"] = s.total; j["width"] = s.width; j["height"] = s.height
    try JSONSerialization.data(withJSONObject: j, options: [.prettyPrinted, .sortedKeys])
        .write(to: evidenceDir.appendingPathComponent(name + ".json"))
}

/// Black linework on white, and nothing else.
func assertBlackOnWhiteDrawing(_ s: Shot) throws {
    try check(s.coloured == 0, "no coloured pixels, got \(s.coloured)")
    try check(s.black >= 500, "black linework present: \(s.black)")
    try check(Double(s.white) > 0.85 * Double(s.total), "mostly white paper: \(s.white)/\(s.total)")
    try check(s.lineFeatures > 0, "queried line features")
    try check(s.textFeatures > 0, "queried text features")
}

@MainActor
func remoteRangeMode(_ window: UIWindow) async throws {
    _ = try await fetch("/__reset")
    let s = try await render(window: window, archiveUrl: store + "/golden.pmtiles", expectContent: true)
    var ranged = 0, bytes = 0
    for e in try await goldenLog() {
        try check(e["status"] as? Int != 200, "a whole-file 200 in Range mode: \(e)")
        try check(e["status"] as? Int == 206, "ranged read: \(e)")
        try check(e["range"] is String, "has a Range header: \(e)")
        ranged += 1
        bytes += e["bytes"] as? Int ?? 0
    }
    try save("remote-range", s, ["rangedRequests": ranged, "rangedBytes": bytes])
    try check(ranged >= 2, "several ranged reads, got \(ranged)")
    try assertBlackOnWhiteDrawing(s)
}

@MainActor
func localFileMode(_ window: UIWindow) async throws {
    _ = try await fetch("/__reset")
    // The app downloads the whole archive itself.
    let (data, resp) = try await fetch("/golden.pmtiles")
    let etag = (resp.value(forHTTPHeaderField: "ETag") ?? "").replacingOccurrences(of: "\"", with: "")
    let hex = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    try check(hex.hasPrefix(etag) && !etag.isEmpty, "downloaded bytes are the archive (etag \(etag))")
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let f = dir.appendingPathComponent("golden.pmtiles")
    try data.write(to: f)
    let before = try await goldenLog()
    try check(before.count == 1, "one download, got \(before.count)")
    try check(before[0]["status"] as? Int == 200, "download was a 200")

    let s = try await render(window: window, archiveUrl: "file://" + f.path, expectContent: true)

    try check(try await goldenLog().count == 1, "opening the local file asked the store for nothing more")
    try save("local-file", s, ["downloadedBytes": data.count, "sha256": hex, "url": "Documents/golden.pmtiles"])
    try assertBlackOnWhiteDrawing(s)
}

/// Control: with no archive, the same checks must see an empty white map, so they can fail.
@MainActor
func missingArchiveRendersBlank(_ window: UIWindow) async throws {
    let s = try await render(window: window, archiveUrl: store + "/nope.pmtiles", expectContent: false)
    try save("missing-archive", s)
    try check(s.lineFeatures == 0 && s.textFeatures == 0, "no features")
    try check(s.black == 0 && s.coloured == 0, "blank")
}

final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    func application(_ application: UIApplication, didFinishLaunchingWithOptions o: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        let w = UIWindow(frame: UIScreen.main.bounds)
        w.rootViewController = UIViewController()
        w.makeKeyAndVisible()
        window = w
        Task { @MainActor in
            var results: [String: Any] = [:]
            let tests: [(String, (UIWindow) async throws -> Void)] = [
                ("remoteRangeMode", remoteRangeMode), ("localFileMode", localFileMode),
                ("missingArchiveRendersBlank", missingArchiveRendersBlank),
            ]
            for (name, run) in tests {
                do { try await run(w); results[name] = "passed" }
                catch let f as Failure { results[name] = "failed: " + f.message }
                catch { results[name] = "failed: \(error)" }
            }
            results["system"] = UIDevice.current.systemVersion
            results["model"] = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] ?? ""
            try? FileManager.default.createDirectory(at: evidenceDir, withIntermediateDirectories: true)
            try? JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
                .write(to: evidenceDir.appendingPathComponent("tests.json"))
            exit(0)
        }
        return true
    }
}

UIApplicationMain(CommandLine.argc, CommandLine.unsafeArgv, nil, NSStringFromClass(AppDelegate.self))
