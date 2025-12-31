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

import android.widget.EditText;
import android.widget.Button;
import android.view.View;
import android.widget.ProgressBar;
import android.text.InputType;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private TextView tv;           // log view (append-only)
    private ScrollView scrollView;

    // New UI elements
    private EditText urlInput;
    private Button loadButton;
    private TextView fileInfo;
    private ProgressBar progressBar;

    private EditText promptInput;
    private Button sendButton;
    private TextView outputView;

    // Llama native instance (field so callbacks can update UI)
    private LlamaNative llama;

    // Model tracking
    private volatile boolean modelLoaded = false;
    private String currentModelPath = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Root layout
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int padding = dpToPx(8);
        root.setPadding(padding, padding, padding, padding);

        // Log area: ScrollView + TextView
        scrollView = new ScrollView(this);
        tv = new TextView(this);
        tv.setText("Starting...\n");
        tv.setTextSize(14);
        int logPadding = dpToPx(12);
        tv.setPadding(logPadding, logPadding, logPadding, logPadding);
        scrollView.addView(tv, new ScrollView.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        // URL input + Load button + file info + progress
        urlInput = new EditText(this);
        urlInput.setHint("Model download URL (https://...)");
        urlInput.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        urlInput.setText("https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf");

        loadButton = new Button(this);
        loadButton.setText("Load Model");

        fileInfo = new TextView(this);
        fileInfo.setText("Model file: (none)");

        progressBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progressBar.setMax(100);
        progressBar.setProgress(0);

        // Prompt input + send button + output
        promptInput = new EditText(this);
        promptInput.setHint("Enter prompt");
        promptInput.setMinLines(2);
        promptInput.setMaxLines(6);

        sendButton = new Button(this);
        sendButton.setText("Send");
        sendButton.setEnabled(false); // disabled until model loaded

        outputView = new TextView(this);
        outputView.setText("Output will appear here");
        outputView.setPadding(logPadding, logPadding, logPadding, logPadding);

        // Add views to root in order
        root.addView(urlInput, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(loadButton, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(fileInfo, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(progressBar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        root.addView(promptInput, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(sendButton, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(outputView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT));

        // Finally add log area at bottom (fill)
        LinearLayout.LayoutParams lpLog = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0);
        lpLog.weight = 1.0f;
        root.addView(scrollView, lpLog);

        setContentView(root);

        appendMessage("UI ready.");

        // Instantiate LlamaNative with overridden onDownloadProgress to update UI
        llama = new LlamaNative() {
            @Override
            public void onDownloadProgress(final int percent) {
                runOnUiThread(() -> {
                    progressBar.setProgress(percent);
                    appendMessage("Download progress: " + percent + "%");
                });
            }
        };

        // Set JNI log path (external files dir)
        File logFile = new File(getExternalFilesDir(null), "ollama.log");
        final String logPath = logFile.getAbsolutePath();
        try {
            llama.setLogPath(logPath);
            appendMessage("Set JNI log path: " + logPath);
        } catch (Throwable t) {
            appendMessage("Failed to call setLogPath(): " + t.getMessage());
        }

        // Load button behavior
        loadButton.setOnClickListener(v -> {
            final String url = urlInput.getText().toString().trim();
            if (url.isEmpty()) {
                showToast("Please enter a download URL");
                return;
            }

            // Extract filename from URL
            final String filename = extractFilenameFromUrl(url);
            if (filename == null || filename.isEmpty()) {
                showToast("Cannot determine filename from URL");
                return;
            }

            final File destFile = new File(getFilesDir(), filename);
            final String modelPath = destFile.getAbsolutePath();

            fileInfo.setText("Model file: " + filename + " (checking...)");
            progressBar.setProgress(0);
            appendMessage("Selected model file: " + modelPath);

            // If exists, skip download and init
            if (destFile.exists() && destFile.length() > 0) {
                appendMessage("Model file already exists: " + destFile.length() + " bytes");
                fileInfo.setText("Model file: " + filename + " (" + destFile.length() + " bytes, exists)");
                // Init model in background
                initModelInBackground(modelPath);
            } else {
                // Download then init
                appendMessage("Starting download: " + url);
                appendMessage("Saving to: " + modelPath);
                new Thread(() -> {
                    String dlResult = null;
                    try {
                        dlResult = llama.download(url, modelPath);
                        appendMessage("download() returned: " + dlResult);
                    } catch (Throwable t) {
                        appendException("download() threw", t);
                        showToast("Download error: " + t.getMessage());
                        return;
                    }

                    if (!"ok".equals(dlResult)) {
                        appendMessage("Download failed: " + dlResult);
                        showToast("Download failed: " + dlResult);
                        return;
                    }

                    File f = new File(modelPath);
                    appendMessage("Model file size: " + f.length() + " bytes");
                    runOnUiThread(() -> fileInfo.setText("Model file: " + filename + " (" + f.length() + " bytes, downloaded)"));

                    // init model
                    initModelInBackground(modelPath);
                }).start();
            }
        });

        // Send button behavior
        sendButton.setOnClickListener(v -> {
            final String userPrompt = promptInput.getText().toString();
            if (userPrompt == null || userPrompt.trim().isEmpty()) {
                showToast("Please enter a prompt");
                return;
            }
            if (!modelLoaded) {
                showToast("Model not loaded yet");
                return;
            }

            // Convert to ChatML template (as original)
            final String chatPrompt = toChatML(userPrompt);

            appendMessage("Running generate...");
            outputView.setText("");
            new Thread(() -> {
                String gen = null;
                try {
                    gen = llama.generate(chatPrompt);
                    final String finalGen = gen;
                    runOnUiThread(() -> {
                        appendMessage("generate() returned.");
                        outputView.setText(finalGen);
                    });
                } catch (Throwable t) {
                    appendException("generate() threw", t);
                    showToast("Generate error: " + t.getMessage());
                }
            }).start();
        });
    }

    private void initModelInBackground(final String modelPath) {
        runOnUiThread(() -> {
            appendMessage("Initializing model...");
            fileInfo.setText("Initializing model...");
            progressBar.setProgress(0);
            loadButton.setEnabled(false);
        });

        new Thread(() -> {
            String initResult = null;
            try {
                initResult = llama.init(modelPath);
            } catch (Throwable t) {
                appendException("init(modelPath) threw", t);
                runOnUiThread(() -> {
                    showToast("Model init error: " + t.getMessage());
                    fileInfo.setText("Model init failed");
                    loadButton.setEnabled(true);
                });
                return;
            }

            // make an effectively-final copy for use in lambdas
            final String finalInitResult = initResult;

            appendMessage("init(modelPath) returned: " + finalInitResult);

            if (!"ok".equals(finalInitResult)) {
                appendMessage("Model init failed: " + finalInitResult);
                runOnUiThread(() -> {
                    showToast("Model init failed: " + finalInitResult);
                    fileInfo.setText("Model init failed: " + finalInitResult);
                    loadButton.setEnabled(true);
                });
                return;
            }

            // Mark loaded
            modelLoaded = true;
            currentModelPath = modelPath;
            runOnUiThread(() -> {
                appendMessage("Model initialized successfully.");
                fileInfo.setText("Model loaded: " + (new File(modelPath).getName()));
                sendButton.setEnabled(true);
                loadButton.setEnabled(true);
                progressBar.setProgress(100);
            });
        }).start();
    }

    // Extract filename from URL (simple heuristic)
    private String extractFilenameFromUrl(String url) {
        if (url == null) return null;
        int q = url.indexOf('?');
        String pure = (q >= 0) ? url.substring(0, q) : url;
        int slash = pure.lastIndexOf('/');
        if (slash >= 0 && slash + 1 < pure.length()) {
            return pure.substring(slash + 1);
        }
        return null;
    }

    // ★ ChatML（EOS 付き）テンプレート
    private String toChatML(String userInput) {
        return "<|system|>\n"
             + "You are a helpful assistant.</s>\n"
             + "<|user|>\n"
             + userInput + "</s>\n"
             + "<|assistant|>\n";
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
