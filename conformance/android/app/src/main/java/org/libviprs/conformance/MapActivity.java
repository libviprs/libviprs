package org.libviprs.conformance;

import android.app.Activity;
import android.os.Bundle;
import android.view.ViewGroup;
import org.maplibre.android.MapLibre;
import org.maplibre.android.maps.MapView;

/** One full-screen MapView. The test drives the style and the camera. */
public class MapActivity extends Activity {
    public MapView mapView;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        MapLibre.getInstance(this);
        mapView = new MapView(this);
        mapView.setLayoutParams(new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(mapView);
        mapView.onCreate(state);
    }

    @Override protected void onStart() { super.onStart(); mapView.onStart(); }
    @Override protected void onResume() { super.onResume(); mapView.onResume(); }
    @Override protected void onPause() { mapView.onPause(); super.onPause(); }
    @Override protected void onStop() { mapView.onStop(); super.onStop(); }
    @Override public void onLowMemory() { super.onLowMemory(); mapView.onLowMemory(); }
    @Override protected void onDestroy() { mapView.onDestroy(); super.onDestroy(); }
}
