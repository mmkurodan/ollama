package com.example.ollama;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.LinearLayout;

import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private TextView tv;
    private ScrollView scrollView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Layout: ScrollView containing a TextView so we can append logs and auto-scroll
        scrollView = new ScrollView(this);
        tv = new TextView(this);
        tv.setText("Starting...\n");
        tv.setTextSize(14);
        int padding = dpToPx(12);
        tv.setPadding(padding, padding, padding, padding);

        scrollView.addView(tv, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.addView(scrollView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(root);

        appendMessage("UI ready.");

        // LlamaNative anonymous subclass to receive progress callbacks
        final LlamaNative llama = new LlamaNative() {
            @Override
            public void onDownloadProgress(final int percent) {
                appendMessage("Download progress: " + percent + "%");
            }
        };

        // Set log path for JNI logging (use app-specific external files dir -> no runtime storage permission required)
        File logFile = new File(getExternalFilesDir(null), "ollama.log");
        final String logPath = logFile.getAbsolutePath();
        try {
            llama.setLogPath(logPath);
            appendMessage("Set JNI log path: " + logPath);
        } catch (Throwable t) {
            appendMessage("Failed to call setLogPath(): " + t.getMessage());
        }

        // Start download+init+generate sequence in background (no pre-init)
        new Thread(() -> {
            try {
                Thread.sleep(100); // allow UI to update
            } catch (InterruptedException ignored) {}

            final String url =
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/"
                + "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf";

            File dir = getFilesDir();
            File modelFile = new File(dir, "tinyllama.gguf");
            final String modelPath = modelFile.getAbsolutePath();

            appendMessage("Starting download: " + url);
            appendMessage("Saving to: " + modelPath);

            String dlResult = null;
            try {
                dlResult = llama.download(url, modelPath);
                appendMessage("download() returned: " + dlResult);
            } catch (Throwable t) {
                appendException("download() threw", t);
                showToast("Download error: " + t.getMessage());
                // stop flow
                runOnUiThread(() -> tv.append("\nOperation aborted due to download exception.\n"));
                return;
            }

            if (!"ok".equals(dlResult)) {
                appendMessage("Download failed: " + dlResult);
                showToast("Download failed: " + dlResult);
                return;
            }

            appendMessage("Model file size: " + modelFile.length() + " bytes");
            appendMessage("Download finished successfully. Calling init(modelPath) to load model...");

            String initResult = null;
            try {
                initResult = llama.init(modelPath);
                appendMessage("init(modelPath) returned: " + initResult);
            } catch (Throwable t) {
                appendException("init(modelPath) threw", t);
                showToast("Model init error: " + t.getMessage());
                return;
            }

            appendMessage("Running test generate(\"Hello!\") ...");
            String gen = null;
            try {
                gen = llama.generate("Hello!");
                appendMessage("generate() returned: " + gen);
            } catch (Throwable t) {
                appendException("generate() threw", t);
                showToast("Generate error: " + t.getMessage());
            }
        }).start();
    }

    private void appendMessage(final String msg) {
        runOnUiThread(() -> {
            tv.append(msg + "\n");
            scrollView.post(() -> scrollView.fullScroll(ScrollView.FOCUS_DOWN));
        });
    }

    private void appendException(final String prefix, final Throwable t) {
        StringWriter sw = new StringWriter();
        t.printStackTrace(new PrintWriter(sw));
        appendMessage(prefix + ": " + t.getMessage());
        appendMessage(sw.toString());
    }

    private void showToast(final String msg) {
        runOnUiThread(() -> {
            Toast toast = Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG);
            toast.setGravity(Gravity.CENTER, 0, 0);
            toast.show();
        });
    }

    private int dpToPx(int dp) {
        float density = getResources().getDisplayMetrics().density;
        return Math.round(dp * density);
    }
}
