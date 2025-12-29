package com.example.ollama;

import android.os.Handler;
import android.os.Looper;
import androidx.annotation.Nullable;

public class LlamaNative {
    static {
        System.loadLibrary("llama_jni");
    }

    // Existing native method declarations (keep/augment as needed by native library)
    // These are examples of native methods that may already exist in the original file.
    // If your repository has different native signatures, they should remain unchanged.
    public static native long cLlamaCreateModel(String path);
    public static native void cLlamaFreeModel(long modelPtr);
    public static native long cLlamaCreateContext(long modelPtr);
    public static native void cLlamaFreeContext(long ctxPtr);
    public static native int cLlamaDownloadModel(String url, String dest);

    // Listener for download progress
    public interface DownloadProgressListener {
        /**
         * Called with a progress value in range [0.0, 1.0].
         */
        void onProgress(float progress);
    }

    private static volatile DownloadProgressListener downloadProgressListener;

    /**
     * Set or clear the download progress listener. The listener will be invoked on the
     * main (UI) thread.
     */
    public static void setDownloadProgressListener(@Nullable DownloadProgressListener listener) {
        downloadProgressListener = listener;
    }

    /**
     * Called from native code to notify about download progress. This method will
     * dispatch the callback to the main/UI thread using a Handler/Looper.
     *
     * The native code should call LlamaNative.onDownloadProgress(progress) where
     * progress is a float between 0.0 and 1.0.
     */
    public static void onDownloadProgress(final float progress) {
        dispatchOnDownloadProgress(progress);
    }

    private static void dispatchOnDownloadProgress(final float progress) {
        final DownloadProgressListener listener = downloadProgressListener;
        if (listener == null) return;

        Handler handler = new Handler(Looper.getMainLooper());
        handler.post(new Runnable() {
            @Override
            public void run() {
                try {
                    listener.onProgress(progress);
                } catch (Throwable t) {
                    // Protect against listener exceptions so native callbacks don't crash the app.
                    t.printStackTrace();
                }
            }
        });
    }
}
