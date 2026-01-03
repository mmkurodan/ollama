package com.example.ollama;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;
import android.view.Gravity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.StringWriter;

import android.widget.EditText;
import android.widget.Button;

import org.json.JSONException;

import java.io.IOException;

public class MainActivity extends Activity {

    private static final String TAG = "MainActivity";
    private static final int REQUEST_SETTINGS = 1;
    
    private TextView logView;           // log view (append-only)
    private ScrollView logScrollView;
    private TextView outputView;
    private ScrollView outputScrollView;

    private EditText promptInput;
    private Button sendButton;
    private Button settingsButton;
    private Button initModelButton;
    private Button viewLogButton;
    private Button clearLogButton;

    // Llama native instance (field so callbacks can update UI)
    private LlamaNative llama;
    
    // Configuration
    private ConfigurationManager configManager;
    private ConfigurationManager.Configuration currentConfig;

    // Model tracking
    private volatile boolean modelLoaded = false;
    private String currentModelPath = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Initialize configuration manager
        configManager = new ConfigurationManager(this);
        
        // Load default configuration
        try {
            currentConfig = configManager.loadConfiguration("default");
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to load default config", e);
            currentConfig = new ConfigurationManager.Configuration();
        }

        // Initialize views from XML
        logView = findViewById(R.id.logView);
        logScrollView = findViewById(R.id.logScrollView);
        outputView = findViewById(R.id.outputView);
        outputScrollView = findViewById(R.id.outputScrollView);
        promptInput = findViewById(R.id.promptInput);
        sendButton = findViewById(R.id.sendButton);
        settingsButton = findViewById(R.id.settingsButton);
        initModelButton = findViewById(R.id.initModelButton);
        viewLogButton = findViewById(R.id.viewLogButton);
        clearLogButton = findViewById(R.id.clearLogButton);

        appendMessage("UI ready.");

        // Instantiate LlamaNative
        llama = new LlamaNative();

        // Set JNI log path (external files dir)
        File logFile = new File(getExternalFilesDir(null), "ollama.log");
        final String logPath = logFile.getAbsolutePath();
        try {
            llama.setLogPath(logPath);
            appendMessage("Set JNI log path: " + logPath);
        } catch (Throwable t) {
            appendMessage("Failed to call setLogPath(): " + t.getMessage());
        }

        // Set up button listeners
        settingsButton.setOnClickListener(v -> openSettings());
        initModelButton.setOnClickListener(v -> reinitializeModel());
        viewLogButton.setOnClickListener(v -> viewLogFile());
        clearLogButton.setOnClickListener(v -> clearLogFile());

        // Send button behavior
        sendButton.setOnClickListener(v -> {
            final String userPrompt = promptInput.getText().toString();
            if (userPrompt == null || userPrompt.trim().isEmpty()) {
                showToast("Please enter a prompt");
                return;
            }
            if (!modelLoaded) {
                showToast("Model not loaded yet. Please load a model in Settings.");
                return;
            }

            // Set parameters before generating
            if (currentConfig != null) {
                try {
                    llama.setParameters(
                        currentConfig.penaltyLastN,
                        (float)currentConfig.penaltyRepeat,
                        (float)currentConfig.penaltyFreq,
                        (float)currentConfig.penaltyPresent,
                        currentConfig.mirostat,
                        (float)currentConfig.mirostatTau,
                        (float)currentConfig.mirostatEta,
                        (float)currentConfig.minP,
                        (float)currentConfig.typicalP,
                        (float)currentConfig.dynatempRange,
                        (float)currentConfig.dynatempExponent,
                        (float)currentConfig.xtcProbability,
                        (float)currentConfig.xtcThreshold,
                        (float)currentConfig.topNSigma,
                        (float)currentConfig.dryMultiplier,
                        (float)currentConfig.dryBase,
                        currentConfig.dryAllowedLength,
                        currentConfig.dryPenaltyLastN,
                        currentConfig.drySequenceBreakers
                    );
                } catch (Throwable t) {
                    Log.e(TAG, "Failed to set parameters", t);
                    appendMessage("Warning: Failed to set parameters: " + t.getMessage());
                }
            }

            // Apply prompt template
            final String chatPrompt = applyPromptTemplate(userPrompt);

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
    
    private void openSettings() {
        Intent intent = new Intent(this, SettingsActivity.class);
        if (currentConfig != null) {
            intent.putExtra(SettingsActivity.EXTRA_CONFIG_NAME, currentConfig.name);
        }
        startActivityForResult(intent, REQUEST_SETTINGS);
    }
    
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_SETTINGS && resultCode == RESULT_OK && data != null) {
            String configName = data.getStringExtra(SettingsActivity.EXTRA_CONFIG_NAME);
            if (configName != null) {
                try {
                    currentConfig = configManager.loadConfiguration(configName);
                    appendMessage("Loaded configuration: " + configName);
                } catch (IOException | JSONException e) {
                    Log.e(TAG, "Failed to load configuration", e);
                    appendMessage("Failed to load configuration: " + e.getMessage());
                }
            }
            
            // Check if a model was loaded in Settings
            String modelPath = data.getStringExtra(SettingsActivity.EXTRA_MODEL_PATH);
            boolean wasModelLoaded = data.getBooleanExtra(SettingsActivity.EXTRA_MODEL_LOADED, false);
            if (modelPath != null && wasModelLoaded) {
                currentModelPath = modelPath;
                modelLoaded = true;
                sendButton.setEnabled(true);
                appendMessage("Model loaded from Settings: " + new File(modelPath).getName());
            }
        }
    }
    
    private void reinitializeModel() {
        if (currentModelPath == null || currentModelPath.isEmpty()) {
            showToast("No model path available. Please load a model in Settings first.");
            return;
        }
        
        appendMessage("Freeing current model...");
        new Thread(() -> {
            try {
                llama.free();
                runOnUiThread(() -> {
                    appendMessage("Model freed.");
                    modelLoaded = false;
                    sendButton.setEnabled(false);
                });
                
                // Small delay to ensure cleanup
                Thread.sleep(500);
                
                // Re-initialize
                appendMessage("Re-initializing model...");
                String initResult = llama.init(currentModelPath);
                
                final String finalInitResult = initResult;
                runOnUiThread(() -> {
                    appendMessage("init() returned: " + finalInitResult);
                    if ("ok".equals(finalInitResult)) {
                        modelLoaded = true;
                        sendButton.setEnabled(true);
                        showToast("Model re-initialized successfully");
                    } else {
                        showToast("Model re-initialization failed: " + finalInitResult);
                    }
                });
            } catch (Throwable t) {
                appendException("Model re-initialization error", t);
                showToast("Error: " + t.getMessage());
            }
        }).start();
    }
    
    private void viewLogFile() {
        File logFile = new File(getExternalFilesDir(null), "ollama.log");
        if (!logFile.exists()) {
            showToast("Log file does not exist");
            return;
        }
        
        new Thread(() -> {
            StringBuilder sb = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(new FileReader(logFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line).append("\n");
                }
                
                final String logContent = sb.toString();
                runOnUiThread(() -> {
                    outputView.setText(logContent);
                    showToast("Displaying log file content");
                });
            } catch (IOException e) {
                Log.e(TAG, "Failed to read log file", e);
                showToast("Failed to read log file: " + e.getMessage());
            }
        }).start();
    }
    
    private void clearLogFile() {
        File logFile = new File(getExternalFilesDir(null), "ollama.log");
        try (FileWriter writer = new FileWriter(logFile, false)) {
            writer.write(""); // Clear the file
            appendMessage("Log file cleared.");
            showToast("Log file cleared");
        } catch (IOException e) {
            Log.e(TAG, "Failed to clear log file", e);
            appendMessage("Failed to clear log file: " + e.getMessage());
            showToast("Failed to clear log file");
        }
    }
    
    private String applyPromptTemplate(String userInput) {
        if (currentConfig == null || currentConfig.promptTemplate == null || currentConfig.promptTemplate.isEmpty()) {
            // Fallback to default template
            return "<|system|>\n"
                 + "You are a helpful assistant.\n"
                 + "<|user|>\n"
                 + userInput + "\n"
                 + "<|assistant|>\n";
        }
        return currentConfig.promptTemplate.replace("{USER_INPUT}", userInput);
    }

    private void appendMessage(final String msg) {
        runOnUiThread(() -> {
            logView.append(msg + "\n");
            logScrollView.post(() -> logScrollView.fullScroll(ScrollView.FOCUS_DOWN));
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
}
