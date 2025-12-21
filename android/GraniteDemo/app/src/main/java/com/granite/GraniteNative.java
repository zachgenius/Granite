package com.granite;

import android.content.res.AssetManager;

public final class GraniteNative {
    static {
        System.loadLibrary("granite_jni");
    }

    private GraniteNative() {}

    public static native String getVersion();

    public static native boolean extractSpirvAssets(
            AssetManager assetManager,
            String assetDir,
            String outputDir);
}
