package com.example.ollama;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private TextView tv;
    private Button btnDownload;

    // Keep a reference so we can clear it later
    private LlamaNative.DownloadProgressListener progressListener;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tv = findViewById(R.id.tv);
        btnDownload = findViewById(R.id.btnDownload);

        btnDownload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startDownload("example-model");
            }
        });
    }

    private void startDownload(final String modelName) {
        // Register download progress listener before starting download
        progressListener = new LlamaNative.DownloadProgressListener() {
            @Override
            public void onProgress(final long downloaded, final long total) {
                // Calculate percent safely
                final int percent = (total > 0) ? (int) ((downloaded * 100) / total) : 0;

                // Update TextView on UI thread
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (tv != null) {
                            tv.setText(percent + "%");
                        }
                    }
                });
            }
        };

        LlamaNative.setDownloadProgressListener(progressListener);

        // Perform download off the UI thread
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Assuming LlamaNative.downloadModel blocks until complete
                    LlamaNative.downloadModel(modelName);

                    // On successful completion ensure UI shows 100%
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (tv != null) tv.setText("100%");
                        }
                    });
                } catch (final Exception e) {
                    Log.e(TAG, "Download failed", e);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (tv != null) tv.setText("Download failed");
                        }
                    });
                } finally {
                    // Clear the listener after download finishes (whether success or failure)
                    try {
                        LlamaNative.setDownloadProgressListener(null);
                    } catch (Exception e) {
                        Log.w(TAG, "Failed to clear download progress listener", e);
                    }
                    progressListener = null;
                }
            }
        }).start();
    }
}
