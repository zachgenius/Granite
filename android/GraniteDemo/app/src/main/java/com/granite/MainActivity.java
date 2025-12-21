package com.granite;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        String cacheDir = getCacheDir().getAbsolutePath() + "/granite_spv";
        GraniteNative.extractSpirvAssets(getAssets(), "granite_spv", cacheDir);

        TextView view = new TextView(this);
        view.setText("Granite version: " + GraniteNative.getVersion());
        setContentView(view);
    }
}
