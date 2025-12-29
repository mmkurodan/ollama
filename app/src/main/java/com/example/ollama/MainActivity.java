package com.example.ollama;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import java.io.PrintWriter;
import java.io.StringWriter;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity";

    private TextView runtimeTextView;
    private Handler mainHandler;
    private LlamaNative llama;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        runtimeTextView = findViewById(R.id.runtimeTextView);
        mainHandler = new Handler(Looper.getMainLooper());

        appendMessage("Activity created");

        // Create an anonymous subclass of LlamaNative to receive progress callbacks
        llama = new LlamaNative() {
            @Override
            public void onDownloadProgress(long downloaded, long total) {
                String progress = String.format("Download progress: %d / %d", downloaded, total);
                appendMessage(progress);
            }

            @Override
            public void onError(Exception e) {
                String err = "LlamaNative error: " + (e == null ? "unknown" : e.getMessage());
                appendMessage(err);
                showToast(err);
                if (e != null) appendException(e);
            }
        };

        // Call preInit on the UI thread
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                appendMessage("Calling preInit on UI thread...");
                try {
                    llama.preInit();
                    appendMessage("preInit completed.");
                } catch (final Exception e) {
                    appendMessage("preInit threw an exception");
                    appendException(e);
                    showToast("preInit failed: " + e.getMessage());
                }
            }
        });

        // Run the download in a background thread to avoid blocking the UI
        new Thread(new Runnable() {
            @Override
            public void run() {
                appendMessage("Starting download in background thread...");
                try {
                    // Replace "model-name" with the actual model identifier as needed.
                    // The exact download method name may vary depending on LlamaNative API.
                    // Common names: downloadModel(...), download(...). Adjust if necessary.
                    llama.downloadModel("model-name");
                    appendMessage("Download completed successfully.");
                } catch (final Exception e) {
                    appendMessage("Download failed with exception");
                    appendException(e);
                    // Show a toast for errors on the UI thread
                    mainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            showToast("Download failed: " + e.getMessage());
                        }
                    });
                }
            }
        }).start();
    }

    private void appendMessage(final String message) {
        Log.d(TAG, message);
        if (mainHandler == null) {
            mainHandler = new Handler(Looper.getMainLooper());
        }
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                if (runtimeTextView != null) {
                    runtimeTextView.append(message + "\n");
                }
            }
        });
    }

    private void appendException(Exception e) {
        StringWriter sw = new StringWriter();
        e.printStackTrace(new PrintWriter(sw));
        appendMessage(sw.toString());
    }

    private void showToast(final String msg) {
        mainHandler.post(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG).show();
            }
        });
    }
}
