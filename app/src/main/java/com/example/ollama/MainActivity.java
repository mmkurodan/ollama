package com.example.ollama;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.Settings;
import android.widget.TextView;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends Activity {

    private static final int REQ_WRITE = 1001;

    private TextView tv;
    private boolean waitingForManagePermission = false;

    private String timestamp() {
        return new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        tv = new TextView(this);
        tv.setText("Starting...");
        tv.setTextSize(16);
        setContentView(tv);

        ensureStoragePermissionOrStart();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (waitingForManagePermission) {
            waitingForManagePermission = false;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                if (Environment.isExternalStorageManager()) {
                    appendLine("[" + timestamp() + "] MANAGE_EXTERNAL_STORAGE granted");
                    startProcess();
                } else {
                    appendLine("[" + timestamp() + "] MANAGE_EXTERNAL_STORAGE not granted");
                }
            }
        }
    }

    private void ensureStoragePermissionOrStart() {
        // For simplicity: request WRITE_EXTERNAL_STORAGE on API < 30; on API >=30 prompt user to grant MANAGE_EXTERNAL_STORAGE if needed.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                appendLine("[" + timestamp() + "] Requesting Manage All Files access via Settings");
                try {
                    Intent intent = new Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION);
                    startActivity(intent);
                    waitingForManagePermission = true;
                } catch (Exception e) {
                    appendLine("[" + timestamp() + "] Failed to launch MANAGE_EXTERNAL_STORAGE settings: " + e.toString());
                    // Fallback: continue without special permission
                    startProcess();
                }
                return;
            }
            // already have manage permission
            startProcess();
        } else {
            // API < 30: request WRITE_EXTERNAL_STORAGE if not granted
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                appendLine("[" + timestamp() + "] Requesting WRITE_EXTERNAL_STORAGE permission");
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE},
                        REQ_WRITE);
            } else {
                startProcess();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_WRITE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                appendLine("[" + timestamp() + "] WRITE_EXTERNAL_STORAGE granted");
                startProcess();
            } else {
                appendLine("[" + timestamp() + "] WRITE_EXTERNAL_STORAGE denied");
                // Proceed anyway, using app internal storage (getFilesDir)
                startProcess();
            }
        }
    }

    private void startProcess() {
        new Thread(() -> {
            appendLine("[" + timestamp() + "] UI ready.");

            String url =
                "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/"
                + "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf";

            File dir = getFilesDir();
            File modelFile = new File(dir, "tinyllama.gguf");
            String modelPath = modelFile.getAbsolutePath();

            appendLine("[" + timestamp() + "] Starting download: " + url);
            appendLine("[" + timestamp() + "] Saving to: " + modelPath);

            LlamaNative llama = new LlamaNative();

            String dlResult = llama.download(url, modelPath);

            if (!"ok".equals(dlResult)) {
                appendLine("[" + timestamp() + "] Download failed: " + dlResult);
                runOnUiThread(() -> tv.setText("Download failed: " + dlResult));
                return;
            }

            appendLine("[" + timestamp() + "] download() returned: " + dlResult);
            appendLine("[" + timestamp() + "] Model file size: " + modelFile.length() + " bytes");
            appendLine("[" + timestamp() + "] Download finished successfully. Calling init(modelPath) to load model...");

            String initResult = llama.init(modelPath);
            appendLine("[" + timestamp() + "] Init result: " + initResult);

            if (!"ok".equals(initResult)) {
                appendLine("[" + timestamp() + "] Init failed: " + initResult);
                runOnUiThread(() -> tv.setText("Init failed: " + initResult));
                return;
            }

            appendLine("[" + timestamp() + "] Starting generate(\"Hello!\")...");
            String gen = llama.generate("Hello!");
            appendLine("[" + timestamp() + "] Generated result length=" + (gen == null ? 0 : gen.length()));
            appendLine("\nGenerated:\n" + gen + "\n");

            runOnUiThread(() -> tv.setText(getHistoryText()));

        }).start();
    }

    private final StringBuilder history = new StringBuilder();

    private void appendLine(String s) {
        synchronized (history) {
            history.append(s).append('\n');
            String text = history.toString();
            runOnUiThread(() -> tv.setText(text));
        }
    }

    private String getHistoryText() {
        synchronized (history) {
            return history.toString();
        }
    }
}
