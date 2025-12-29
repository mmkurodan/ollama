package com.example.ollama;

import android.util.Log;

public class LlamaNative {

    private static final String TAG = "LlamaNative";

    static {
        System.loadLibrary("llama_jni");
    }

    public native String download(String url, String path);
    public native String init(String modelPath);
    public native String generate(String prompt);

    // Called from native code to deliver download progress (0-100)
    // Implement UI dispatching here if needed (e.g. post to main thread)
    public void onDownloadProgress(int percent) {
        Log.d(TAG, "Download progress: " + percent + "%");
    }
}
