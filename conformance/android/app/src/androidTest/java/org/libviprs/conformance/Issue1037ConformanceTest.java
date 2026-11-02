package org.libviprs.conformance;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.security.MessageDigest;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.maplibre.android.camera.CameraPosition;
import org.maplibre.android.geometry.LatLng;
import org.maplibre.android.maps.MapLibreMap;
import org.maplibre.android.maps.Style;

/**
 * Issue #1037, Android half: the golden PMTiles, drawn by MapLibre Native Android with the shared
 * black on white style. The server is web/vector-viewer/e2e/server.mjs, reached from the emulator
 * at 10.0.2.2 (the host's loopback).
 */
@RunWith(AndroidJUnit4.class)
public class Issue1037ConformanceTest {
    static final String STORE = "http://10.0.2.2:4174";
    static final String GLYPHS = "http://10.0.2.2:4173/fonts/{fontstack}/{range}.pbf";

    /** What one render produced. */
    static final class Shot {
        int lineFeatures, textFeatures, black, white, other, colored, total;
        Bitmap bitmap;
    }

    final Context ctx = InstrumentationRegistry.getInstrumentation().getTargetContext();

    /** The shared style with its two placeholders filled, as the Rust helper does. */
    String styleFor(String archiveUrl) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (InputStream in = ctx.getAssets().open("black_on_white.style.json")) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
            sb.append(out.toString("UTF-8"));
        }
        JSONObject style = new JSONObject(sb.toString());
        style.getJSONObject("sources").getJSONObject("drawing").put("url", "pmtiles://" + archiveUrl);
        style.put("glyphs", GLYPHS);
        return style.toString();
    }

    static String get(String url) throws Exception {
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        try (InputStream in = c.getInputStream()) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
            return out.toString("UTF-8");
        }
    }

    /** Requests the store saw for the golden archive since the last reset. */
    JSONArray goldenLog() throws Exception {
        JSONArray all = new JSONArray(get(STORE + "/__log")), mine = new JSONArray();
        for (int i = 0; i < all.length(); i++) {
            if (all.getJSONObject(i).getString("path").endsWith("/golden.pmtiles")) mine.put(all.get(i));
        }
        return mine;
    }

    /** Loads the style in a fresh map at a fixed camera, waits for drawing, and measures it. */
    Shot render(String archiveUrl, boolean expectContent) throws Exception {
        String json = styleFor(archiveUrl);
        Shot shot = new Shot();
        AtomicReference<MapLibreMap> mapRef = new AtomicReference<>();
        CountDownLatch styled = new CountDownLatch(1);
        try (ActivityScenario<MapActivity> sc = ActivityScenario.launch(MapActivity.class)) {
            sc.onActivity(a -> a.mapView.getMapAsync(map -> {
                map.getUiSettings().setAttributionEnabled(false);
                map.getUiSettings().setLogoEnabled(false);
                map.getUiSettings().setCompassEnabled(false);
                // Known camera: the middle of the drawing, zoom 1.5.
                map.setCameraPosition(new CameraPosition.Builder().target(new LatLng(0, 0)).zoom(1.5).build());
                mapRef.set(map);
                map.setStyle(new Style.Builder().fromJson(json), s -> styled.countDown());
            }));
            assertTrue("style did not load", styled.await(60, TimeUnit.SECONDS));
            MapLibreMap map = mapRef.get();
            float[] size = new float[2];
            InstrumentationRegistry.getInstrumentation().runOnMainSync(() -> {
                size[0] = map.getWidth();
                size[1] = map.getHeight();
            });
            RectF all = new RectF(0, 0, size[0], size[1]);
            long deadline = System.currentTimeMillis() + (expectContent ? 90_000 : 12_000);
            while (System.currentTimeMillis() < deadline) {
                Thread.sleep(500);
                InstrumentationRegistry.getInstrumentation().runOnMainSync(() -> {
                    shot.lineFeatures = map.queryRenderedFeatures(all, "linework").size();
                    shot.textFeatures = map.queryRenderedFeatures(all, "annotations").size();
                });
                if (expectContent && shot.lineFeatures > 0 && shot.textFeatures > 0) {
                    Thread.sleep(3000); // let the rest of the tiles and the glyphs land
                    break;
                }
            }
            InstrumentationRegistry.getInstrumentation().runOnMainSync(() -> {
                shot.lineFeatures = map.queryRenderedFeatures(all, "linework").size();
                shot.textFeatures = map.queryRenderedFeatures(all, "annotations").size();
            });
            CountDownLatch snapped = new CountDownLatch(1);
            InstrumentationRegistry.getInstrumentation().runOnMainSync(() -> map.snapshot(b -> {
                shot.bitmap = b;
                snapped.countDown();
            }));
            assertTrue("no snapshot", snapped.await(30, TimeUnit.SECONDS));
        }
        measure(shot);
        return shot;
    }

    static void measure(Shot s) {
        int w = s.bitmap.getWidth(), h = s.bitmap.getHeight();
        int[] px = new int[w * h];
        s.bitmap.getPixels(px, 0, w, 0, 0, w, h);
        s.total = px.length;
        for (int p : px) {
            int r = (p >> 16) & 0xff, g = (p >> 8) & 0xff, b = p & 0xff;
            int max = Math.max(r, Math.max(g, b)), min = Math.min(r, Math.min(g, b));
            if (max - min > 24) s.colored++;
            else if (max <= 64) s.black++;
            else if (min >= 245) s.white++;
            else s.other++;
        }
    }

    void save(String name, Shot s, JSONObject extra) throws Exception {
        File dir = new File(ctx.getExternalFilesDir(null), "evidence");
        assertTrue(dir.isDirectory() || dir.mkdirs());
        try (FileOutputStream out = new FileOutputStream(new File(dir, name + ".png"))) {
            s.bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
        }
        JSONObject j = extra == null ? new JSONObject() : extra;
        j.put("lineFeatures", s.lineFeatures).put("textFeatures", s.textFeatures);
        j.put("blackPx", s.black).put("whitePx", s.white).put("greyPx", s.other).put("colouredPx", s.colored);
        j.put("totalPx", s.total).put("width", s.bitmap.getWidth()).put("height", s.bitmap.getHeight());
        try (FileOutputStream out = new FileOutputStream(new File(dir, name + ".json"))) {
            out.write(j.toString(2).getBytes("UTF-8"));
        }
    }

    /** Black linework on white, and nothing else. */
    static void assertBlackOnWhiteDrawing(Shot s) {
        assertEquals("no coloured pixels", 0, s.colored);
        assertTrue("black linework present: " + s.black, s.black >= 500);
        assertTrue("mostly white paper: " + s.white + "/" + s.total, s.white > 0.85 * s.total);
        assertTrue("queried line features", s.lineFeatures > 0);
        assertTrue("queried text features", s.textFeatures > 0);
    }

    @Test
    public void remoteRangeMode() throws Exception {
        get(STORE + "/__reset");
        Shot s = render(STORE + "/golden.pmtiles", true);
        JSONArray log = goldenLog();
        long ranged = 0, bytes = 0;
        for (int i = 0; i < log.length(); i++) {
            JSONObject e = log.getJSONObject(i);
            assertFalse("a whole-file 200 in Range mode: " + e, e.getInt("status") == 200);
            assertEquals("ranged read: " + e, 206, e.getInt("status"));
            assertTrue("has a Range header: " + e, e.has("range") && !e.isNull("range"));
            ranged++;
            bytes += e.getLong("bytes");
        }
        assertTrue("several ranged reads, got " + ranged, ranged >= 2);
        save("remote-range", s, new JSONObject().put("rangedRequests", ranged).put("rangedBytes", bytes));
        assertBlackOnWhiteDrawing(s);
    }

    @Test
    public void localFileMode() throws Exception {
        get(STORE + "/__reset");
        // The app downloads the whole archive itself.
        File f = new File(ctx.getFilesDir(), "golden.pmtiles");
        HttpURLConnection c = (HttpURLConnection) new URL(STORE + "/golden.pmtiles").openConnection();
        String etag = c.getHeaderField("ETag").replace("\"", "");
        MessageDigest sha = MessageDigest.getInstance("SHA-256");
        long len = 0;
        try (InputStream in = c.getInputStream(); FileOutputStream out = new FileOutputStream(f)) {
            byte[] buf = new byte[8192];
            for (int n; (n = in.read(buf)) > 0; ) {
                out.write(buf, 0, n);
                sha.update(buf, 0, n);
                len += n;
            }
        }
        StringBuilder hex = new StringBuilder();
        for (byte b : sha.digest()) hex.append(String.format("%02x", b));
        assertTrue("downloaded bytes are the archive (etag " + etag + ")", hex.toString().startsWith(etag));
        assertEquals(len, f.length());

        JSONArray before = goldenLog();
        assertEquals("one download", 1, before.length());
        assertEquals(200, before.getJSONObject(0).getInt("status"));

        Shot s = render("file://" + f.getAbsolutePath(), true);

        assertEquals("opening the local file asked the store for nothing more", 1, goldenLog().length());
        save("local-file", s, new JSONObject().put("downloadedBytes", len).put("sha256", hex.toString())
                .put("url", "pmtiles://file://" + f.getAbsolutePath()));
        assertBlackOnWhiteDrawing(s);
    }

    /** Control: with no archive, the same checks must see an empty white map, so they can fail. */
    @Test
    public void missingArchiveRendersBlank() throws Exception {
        Shot s = render(STORE + "/nope.pmtiles", false);
        save("missing-archive", s, null);
        assertEquals(0, s.lineFeatures);
        assertEquals(0, s.textFeatures);
        assertEquals(0, s.black);
        assertEquals(0, s.colored);
    }
}
