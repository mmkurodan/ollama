package com.micklab.llama;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.ActivityNotFoundException;
import android.content.ContentValues;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Typeface;
import android.net.ConnectivityManager;
import android.net.LinkAddress;
import android.net.LinkProperties;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.Uri;
import android.os.PowerManager;
import android.provider.Settings;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.AutoCompleteTextView;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.CompoundButton;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONException;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.micklab.llama.ConfigurationManager.Configuration.DEFAULT_DRY_SEQUENCE_BREAKERS;

public class SettingsActivity extends Activity {
    private static final String TAG = "SettingsActivity";
    
    public static final String EXTRA_CONFIG_NAME = "config_name";
    public static final String EXTRA_MODEL_PATH = "model_path";
    public static final String EXTRA_MODEL_LOADED = "model_loaded";
    public static final String EXTRA_API_PORT = "api_port";
    public static final String EXTRA_DISPLAY_LANGUAGE = "display_language";
    
    private static final String PREFS_NAME = "ollama_prefs";
    private static final String PREF_API_PORT = "api_port";
    private static final String PREF_LOG_LEVEL = "log_level";
    private static final String PREF_SHOW_PERF_METRICS = "show_perf_metrics";
    // Busy-queue max wait (seconds) a queued API request waits for the model slot. Must match
    // OllamaApiServer's key. 0 = unlimited (wait forever); otherwise clamped to [30, 600].
    private static final String PREF_BUSY_QUEUE_WAIT_SECONDS = "busy_queue_wait_seconds";
    private static final int DEFAULT_BUSY_QUEUE_WAIT_SECONDS = 600;
    private static final int BUSY_QUEUE_WAIT_MIN_SECONDS = 30;
    private static final int BUSY_QUEUE_WAIT_MAX_SECONDS = 600;
    private static final int REQUEST_IMPORT_MODEL_LOCAL_DEVICE = 1001;
    private static final int REQUEST_RESTORE_DIR = 1002;
    private static final int REQUEST_IMPORT_LORA_ADAPTER = 1003;
    private static final int MODEL_COPY_BUFFER_SIZE = 1024 * 1024;
    private static final String IMPORT_TEMP_SUFFIX = ".import.tmp";
    
    private ConfigurationManager configManager;
    private ModelManager modelManager;
    
    // UI elements
    private AutoCompleteTextView configNameInput;
    private EditText modelUrlInput;
    private TextView multimodalProjectorInfo;
    private Button selectProjectorButton;
    private Button clearProjectorButton;
    private Button applyLoraButton;
    private Button clearLoraButton;
    private TextView loraAdapterInfo;
    private Button mtpModelButton;
    private Switch mtpEnableToggle;
    private EditText mtpNDraftInput;
    private Button searchGgufButton;
    private EditText nCtxInput;
    private EditText nThreadsInput;
    private EditText nBatchInput;
    private EditText tempInput;
    private EditText topPInput;
    private EditText topKInput;
    private TextView autoSelectedTemplateView;
    private TextView modelFileInfo;
    private ProgressBar modelProgressBar;
    private Button importModelButton;
    private Button loadModelButton;
    private Button selectModelButton;
    private Button maintainModelButton;
    
    // Penalty parameter inputs
    private EditText penaltyLastNInput;
    private EditText penaltyRepeatInput;
    private EditText penaltyFreqInput;
    private EditText penaltyPresentInput;
    
    // Mirostat parameter inputs
    private EditText mirostatInput;
    private EditText mirostatTauInput;
    private EditText mirostatEtaInput;
    
    // Additional sampling parameter inputs
    private EditText minPInput;
    private EditText typicalPInput;
    private EditText dynatempRangeInput;
    private EditText dynatempExponentInput;
    private EditText xtcProbabilityInput;
    private EditText xtcThresholdInput;
    private EditText topNSigmaInput;
    
    // DRY parameter inputs
    private EditText dryMultiplierInput;
    private EditText dryBaseInput;
    private EditText dryAllowedLengthInput;
    private EditText dryPenaltyLastNInput;
    private EditText drySequenceBreakersInput;
    
    // Max output tokens
    private EditText nPredictInput;

    // KV cache quantization
    private Spinner kvCacheTypeSpinner;
    private static final int[] KV_CACHE_TYPE_IDS = { 1, 8, 7, 6, 3, 2, 20 };
    private static final String[] KV_CACHE_TYPE_NAMES = { "F16 (default)", "Q8.0", "Q5.1", "Q5.0", "Q4.1", "Q4.0", "IQ4_NL" };

    // GPU switch stabilization
    private Spinner gpuStabSpinner;

    // Runtime switches
    private Switch streamingSwitch;
    private Switch showPerfMetricsSwitch;
    private SeekBar gpuLayersSeekBar;
    private TextView gpuLayersValue;
    private SeekBar busyTimeoutSeekBar;
    private TextView busyTimeoutValue;
    private Switch keepAwakeSwitch;
    private Switch recoveryWatchdogSwitch;
    private Switch recoveryRecycleSwitch;
    private EditText recoveryRecycleIntervalInput;
    private Button batteryOptButton;
    private Switch enableThinkingSwitch;
    private Switch useMmapSwitch;
    private Switch nCtxAutoExpandSwitch;
    // Compute backend (backendType is derived from this switch; off = CPU, on = GPU)
    private Switch  gpuEnabledSwitch;

    // New prompt settings
    private EditText systemPromptInput;
    private EditText customChatTemplateInput;
    
    // API Server settings
    private EditText apiPortInput;
    private TextView apiLoopbackUrlView;
    private LinearLayout apiWifiUrlContainer;
    private TextView apiWifiUrlView;
    private TextView apiWifiUrlHintView;
    private EditText mcpConfigJsonInput;
    private EditText functionDefinitionsJsonInput;
    private Switch sharedMcpEnabledSwitch;
    private Switch sharedFunctionCallingEnabledSwitch;
    private TextView languageLabel;
    private Spinner languageSpinner;
    private Spinner logLevelSpinner;
    private Button licenseButton;
    private Button documentsButton;
    private Button saveConfigButton;
    private Button loadConfigButton;
    private Button deleteConfigButton;
    private Button backupProfilesButton;
    private Button restoreProfilesButton;
    private Button backButton;
    private Button cancelButton;
    private LinearLayout settingsContentContainer;
    
    private ConfigurationManager.Configuration currentConfig;
    private ArrayAdapter<String> configAdapter;
    private String loadedModelPath = null;
    private String selectedProjectorReference = "";
    private String selectedMtpReference = "";   // MTP draft source ("" = model's own head)
    private boolean selectedProjectorManualSelection = false;
    // True when the user explicitly tapped "Clear Projector": disables mmproj auto-discovery
    // so the clear actually sticks (see ModelManager.resolveMultimodalProjectorPath).
    private boolean selectedProjectorDisabled = false;
    private boolean modelLoadedSuccessfully = false;
    private volatile boolean importInProgress = false;
    private volatile boolean huggingFaceSearchInProgress = false;
    // Quantization chosen in the search dialog. It is a per-file property, so rather than filtering
    // repositories it narrows the GGUF file list shown for the selected repository. Empty = no filter.
    private String huggingFaceQuantFilter = "";
    private final List<CollapsibleSectionController> collapsibleSections = new ArrayList<>();
    private final ModelManager.BusyStateListener busyStateListener =
            busy -> runOnUiThread(this::updateActionButtonStateForBusy);
    private final ConnectivityManager.NetworkCallback networkCallback = new ConnectivityManager.NetworkCallback() {
        @Override
        public void onAvailable(Network network) {
            runOnUiThread(SettingsActivity.this::updateApiServerUrlViews);
        }

        @Override
        public void onLost(Network network) {
            runOnUiThread(SettingsActivity.this::updateApiServerUrlViews);
        }

        @Override
        public void onCapabilitiesChanged(Network network, NetworkCapabilities networkCapabilities) {
            runOnUiThread(SettingsActivity.this::updateApiServerUrlViews);
        }

        @Override
        public void onLinkPropertiesChanged(Network network, LinkProperties linkProperties) {
            runOnUiThread(SettingsActivity.this::updateApiServerUrlViews);
        }
    };
    private volatile int lastDownloadProgress = 0;
    private ConnectivityManager connectivityManager;
    private boolean networkCallbackRegistered = false;

    private static final class ImportedModelCandidate {
        private final String displayName;
        private final long sizeBytes;

        private ImportedModelCandidate(String displayName, long sizeBytes) {
            this.displayName = displayName;
            this.sizeBytes = sizeBytes;
        }
    }

    private interface SelectionHandler {
        void onSelected(int selectedIndex);
    }

    private static final class CollapsibleSectionController {
        private final TextView indicatorView;
        private final View contentView;
        private boolean expanded;

        private CollapsibleSectionController(TextView indicatorView, View contentView) {
            this.indicatorView = indicatorView;
            this.contentView = contentView;
        }
    }

    @Override
    protected void attachBaseContext(Context newBase) {
        super.attachBaseContext(AppLanguageManager.wrap(newBase));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_settings);
        
        configManager = new ConfigurationManager(this);
        
        // Get ModelManager singleton
        modelManager = ModelManager.getInstance(this);
        
        initViews();
        modelManager.addBusyStateListener(busyStateListener);

        // Register download progress listener after views are initialized to avoid NPE if native
        // code emits progress immediately.
        modelManager.getLlama().setDownloadProgressListener(percent -> {
            if (percent == lastDownloadProgress) {
                return;
            }
            lastDownloadProgress = percent;
            runOnUiThread(() -> {
                modelProgressBar.setProgress(percent);
                modelFileInfo.setText(localizedText("モデルをダウンロード中... ", "Downloading model... ") + percent + "%");
            });
        });
        
        loadConfigList();
        
        // Load configuration from intent or default
        String configName = getIntent().getStringExtra(EXTRA_CONFIG_NAME);
        if (configName == null || configName.isEmpty()) {
            configName = "default";
        }
        loadConfigurationByName(configName);
    }
    
    private void initViews() {
        configNameInput = findViewById(R.id.configNameInput);
        modelUrlInput = findViewById(R.id.modelUrlInput);
        multimodalProjectorInfo = findViewById(R.id.multimodalProjectorInfo);
        selectProjectorButton = findViewById(R.id.selectProjectorButton);
        clearProjectorButton = findViewById(R.id.clearProjectorButton);
        applyLoraButton = findViewById(R.id.applyLoraButton);
        clearLoraButton = findViewById(R.id.clearLoraButton);
        loraAdapterInfo = findViewById(R.id.loraAdapterInfo);
        mtpModelButton = findViewById(R.id.mtpModelButton);
        mtpEnableToggle = findViewById(R.id.mtpEnableToggle);
        mtpNDraftInput = findViewById(R.id.mtpNDraftInput);
        searchGgufButton = findViewById(R.id.searchGgufButton);
        nCtxInput = findViewById(R.id.nCtxInput);
        nCtxAutoExpandSwitch = findViewById(R.id.nCtxAutoExpandSwitch);
        nThreadsInput = findViewById(R.id.nThreadsInput);
        nBatchInput = findViewById(R.id.nBatchInput);
        tempInput = findViewById(R.id.tempInput);
        topPInput = findViewById(R.id.topPInput);
        topKInput = findViewById(R.id.topKInput);
        autoSelectedTemplateView = findViewById(R.id.autoSelectedTemplateView);
        modelFileInfo = findViewById(R.id.modelFileInfo);
        modelProgressBar = findViewById(R.id.modelProgressBar);
        importModelButton = findViewById(R.id.importModelButton);
        loadModelButton = findViewById(R.id.loadModelButton);
        selectModelButton = findViewById(R.id.selectModelButton);
        maintainModelButton = findViewById(R.id.maintainModelButton);
        
        // Penalty parameter inputs
        penaltyLastNInput = findViewById(R.id.penaltyLastNInput);
        penaltyRepeatInput = findViewById(R.id.penaltyRepeatInput);
        penaltyFreqInput = findViewById(R.id.penaltyFreqInput);
        penaltyPresentInput = findViewById(R.id.penaltyPresentInput);
        
        // Mirostat parameter inputs
        mirostatInput = findViewById(R.id.mirostatInput);
        mirostatTauInput = findViewById(R.id.mirostatTauInput);
        mirostatEtaInput = findViewById(R.id.mirostatEtaInput);
        
        // Additional sampling parameter inputs
        minPInput = findViewById(R.id.minPInput);
        typicalPInput = findViewById(R.id.typicalPInput);
        dynatempRangeInput = findViewById(R.id.dynatempRangeInput);
        dynatempExponentInput = findViewById(R.id.dynatempExponentInput);
        xtcProbabilityInput = findViewById(R.id.xtcProbabilityInput);
        xtcThresholdInput = findViewById(R.id.xtcThresholdInput);
        topNSigmaInput = findViewById(R.id.topNSigmaInput);
        
        // DRY parameter inputs
        dryMultiplierInput = findViewById(R.id.dryMultiplierInput);
        dryBaseInput = findViewById(R.id.dryBaseInput);
        dryAllowedLengthInput = findViewById(R.id.dryAllowedLengthInput);
        dryPenaltyLastNInput = findViewById(R.id.dryPenaltyLastNInput);
        drySequenceBreakersInput = findViewById(R.id.drySequenceBreakersInput);
        
        // Max output tokens
        nPredictInput = findViewById(R.id.nPredictInput);

        // KV cache quantization spinner
        kvCacheTypeSpinner = findViewById(R.id.kvCacheTypeSpinner);
        if (kvCacheTypeSpinner != null) {
            ArrayAdapter<String> kvAdapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_spinner_item, KV_CACHE_TYPE_NAMES);
            kvAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            kvCacheTypeSpinner.setAdapter(kvAdapter);
        }

        // GPU switch stabilization spinner
        gpuStabSpinner = findViewById(R.id.gpuStabSpinner);
        if (gpuStabSpinner != null) {
            String[] stabLabels = {
                localizedText("Off（無効）", "Off"),
                localizedText("Low（200ms）", "Low (200ms)"),
                localizedText("Medium（500ms）", "Medium (500ms)"),
                localizedText("High（1000ms）", "High (1000ms)")
            };
            ArrayAdapter<String> stabAdapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_spinner_item, stabLabels);
            stabAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
            gpuStabSpinner.setAdapter(stabAdapter);
        }

        // Streaming switch
        streamingSwitch = findViewById(R.id.streamingSwitch);
        showPerfMetricsSwitch = findViewById(R.id.showPerfMetricsSwitch);

        // Compute backend: GPU トグル (OFF = CPU, ON = GPU)。backendType をこれから導出する。
        gpuEnabledSwitch = findViewById(R.id.gpuEnabledSwitch);
        CompoundButton.OnCheckedChangeListener backendToggleListener = (button, checked) -> {
            if (checked) {
                // GPU を有効化したとき、オフロード未設定(0)なら ALL(-1) を既定値にする
                if (gpuLayersSeekBar != null && gpuLayersSeekBar.getProgress() == 0) {
                    gpuLayersSeekBar.setProgress(40); // 40 = ALL (= -1)
                }
            }
            updateBackendDependentUi();
        };
        if (gpuEnabledSwitch != null) gpuEnabledSwitch.setOnCheckedChangeListener(backendToggleListener);

        gpuLayersSeekBar = findViewById(R.id.gpuLayersSeekBar);
        gpuLayersValue = findViewById(R.id.gpuLayersValue);
        gpuLayersSeekBar.setMax(40);
        gpuLayersSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                gpuLayersValue.setText(progress > 39 ? "ALL" : String.valueOf(progress));
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Busy-queue wait timeout (how long a queued API request waits for the model slot).
        // SeekBar 0..600: 0 = unlimited (wait forever); 1..29 snap up to the 30s minimum.
        TextView busyTimeoutLabel = findViewById(R.id.busyTimeoutLabel);
        if (busyTimeoutLabel != null) {
            busyTimeoutLabel.setText(localizedText(
                    "待機中リクエストのタイムアウト:", "Busy wait timeout for queued requests:"));
        }
        busyTimeoutSeekBar = findViewById(R.id.busyTimeoutSeekBar);
        busyTimeoutValue = findViewById(R.id.busyTimeoutValue);
        if (busyTimeoutSeekBar != null) {
            busyTimeoutSeekBar.setMax(BUSY_QUEUE_WAIT_MAX_SECONDS);
            busyTimeoutSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    if (busyTimeoutValue != null) {
                        busyTimeoutValue.setText(formatBusyTimeoutLabel(progress));
                    }
                }
                @Override public void onStartTrackingTouch(SeekBar seekBar) {}
                @Override public void onStopTrackingTouch(SeekBar seekBar) {
                    int p = seekBar.getProgress();
                    if (p > 0 && p < BUSY_QUEUE_WAIT_MIN_SECONDS) {
                        seekBar.setProgress(BUSY_QUEUE_WAIT_MIN_SECONDS);
                    }
                }
            });
            int storedBusySeconds = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
                    .getInt(PREF_BUSY_QUEUE_WAIT_SECONDS, DEFAULT_BUSY_QUEUE_WAIT_SECONDS);
            int initialProgress = storedBusySeconds <= 0
                    ? 0
                    : Math.min(BUSY_QUEUE_WAIT_MAX_SECONDS, Math.max(BUSY_QUEUE_WAIT_MIN_SECONDS, storedBusySeconds));
            busyTimeoutSeekBar.setProgress(initialProgress);
            if (busyTimeoutValue != null) {
                busyTimeoutValue.setText(formatBusyTimeoutLabel(initialProgress));
            }
        }

        // Background execution: keep-awake toggle + battery-optimization exemption.
        keepAwakeSwitch = findViewById(R.id.keepAwakeSwitch);
        if (keepAwakeSwitch != null) {
            keepAwakeSwitch.setText(localizedText(
                    "スリープ時も稼働（電池消費大）",
                    "Run during sleep (more battery)"));
            keepAwakeSwitch.setChecked(OllamaForegroundService.isKeepAwakeEnabled(this));
        }
        batteryOptButton = findViewById(R.id.batteryOptButton);
        if (batteryOptButton != null) {
            batteryOptButton.setText(localizedText("電池最適化を無効化", "Ignore battery optimization"));
            batteryOptButton.setOnClickListener(v -> requestIgnoreBatteryOptimizations());
        }

        // Self-recovery: watchdog (auto-restart after a kill) + optional proactive process recycle.
        recoveryWatchdogSwitch = findViewById(R.id.recoveryWatchdogSwitch);
        if (recoveryWatchdogSwitch != null) {
            recoveryWatchdogSwitch.setText(localizedText(
                    "強制終了時の自動再起動",
                    "Auto-restart if killed"));
            recoveryWatchdogSwitch.setChecked(RecoveryScheduler.isWatchdogEnabled(this));
        }
        recoveryRecycleSwitch = findViewById(R.id.recoveryRecycleSwitch);
        if (recoveryRecycleSwitch != null) {
            recoveryRecycleSwitch.setText(localizedText(
                    "OEM強制終了予防の定期再起動",
                    "Periodic restart (avoid OEM force-stop)"));
            recoveryRecycleSwitch.setChecked(RecoveryScheduler.isRecycleEnabled(this));
        }
        TextView recoveryRecycleIntervalLabel = findViewById(R.id.recoveryRecycleIntervalLabel);
        if (recoveryRecycleIntervalLabel != null) {
            recoveryRecycleIntervalLabel.setText(localizedText("再起動の間隔（分）:", "Recycle interval (min):"));
        }
        recoveryRecycleIntervalInput = findViewById(R.id.recoveryRecycleIntervalInput);
        if (recoveryRecycleIntervalInput != null) {
            recoveryRecycleIntervalInput.setText(String.valueOf(RecoveryScheduler.getRecycleIntervalMinutes(this)));
        }
        enableThinkingSwitch = findViewById(R.id.enableThinkingSwitch);
        useMmapSwitch = findViewById(R.id.useMmapSwitch);

        // New prompt settings
        systemPromptInput = findViewById(R.id.systemPromptInput);
        customChatTemplateInput = findViewById(R.id.customChatTemplateInput);
        
        // API Server settings
        apiPortInput = findViewById(R.id.apiPortInput);
        apiLoopbackUrlView = findViewById(R.id.apiLoopbackUrlView);
        apiWifiUrlContainer = findViewById(R.id.apiWifiUrlContainer);
        apiWifiUrlView = findViewById(R.id.apiWifiUrlView);
        apiWifiUrlHintView = findViewById(R.id.apiWifiUrlHintView);
        mcpConfigJsonInput = findViewById(R.id.mcpConfigJsonInput);
        functionDefinitionsJsonInput = findViewById(R.id.functionDefinitionsJsonInput);
        sharedMcpEnabledSwitch = findViewById(R.id.sharedMcpEnabledSwitch);
        sharedFunctionCallingEnabledSwitch = findViewById(R.id.sharedFunctionCallingEnabledSwitch);
        languageLabel = findViewById(R.id.languageLabel);
        languageSpinner = findViewById(R.id.languageSpinner);
        logLevelSpinner = findViewById(R.id.logLevelSpinner);
        settingsContentContainer = findViewById(R.id.settingsContentContainer);
        licenseButton = findViewById(R.id.licenseButton);
        documentsButton = findViewById(R.id.documentsButton);
        connectivityManager = getSystemService(ConnectivityManager.class);
        
        // Load saved API port
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        int savedPort = prefs.getInt(PREF_API_PORT, OllamaApiServer.DEFAULT_PORT);
        apiPortInput.setText(String.valueOf(savedPort));
        apiPortInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                apiPortInput.setError(null);
                updateApiServerUrlViews();
            }
        });
        mcpConfigJsonInput.setText(McpSettingsHelper.getSharedMcpServersJson(this));
        mcpConfigJsonInput.setHint(McpSettingsHelper.getSharedMcpServersHint());
        functionDefinitionsJsonInput.setText(McpSettingsHelper.getSharedFunctionDefinitionsJson(this));
        functionDefinitionsJsonInput.setHint(McpSettingsHelper.getSharedFunctionDefinitionsHint());
        sharedMcpEnabledSwitch.setChecked(McpSettingsHelper.isSharedMcpEnabledOutsideWebUi(this));
        sharedFunctionCallingEnabledSwitch.setChecked(
                McpSettingsHelper.isSharedFunctionDefinitionsEnabledOutsideWebUi(this));
        int defaultLogLevel = 2;
        int savedLogLevel = prefs.contains(PREF_LOG_LEVEL)
                ? prefs.getInt(PREF_LOG_LEVEL, defaultLogLevel)
                : defaultLogLevel;
        setupLanguageSpinner();
        setupLogLevelSpinner(savedLogLevel);
        
        saveConfigButton = findViewById(R.id.saveConfigButton);
        loadConfigButton = findViewById(R.id.loadConfigButton);
        deleteConfigButton = findViewById(R.id.deleteConfigButton);
        backupProfilesButton = findViewById(R.id.backupProfilesButton);
        restoreProfilesButton = findViewById(R.id.restoreProfilesButton);
        backButton = findViewById(R.id.backButton);
        cancelButton = findViewById(R.id.cancelButton);

        saveConfigButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) return;
            String name = configNameInput.getText().toString().trim();
            new AlertDialog.Builder(this)
                .setTitle(localizedText("設定を保存", "Save Configuration"))
                .setMessage(localizedText("「" + name + "」を保存しますか？", "Save \"" + name + "\"?"))
                .setPositiveButton(localizedText("保存", "Save"), (d, w) -> saveCurrentConfiguration())
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
        });
        loadConfigButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            loadSelectedConfiguration();
        });
        deleteConfigButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) return;
            String name = configNameInput.getText().toString().trim();
            if (name.isEmpty()) {
                showToast(localizedText("設定が選択されていません", "No configuration selected"));
                return;
            }
            new AlertDialog.Builder(this)
                .setTitle(localizedText("設定を削除", "Delete Configuration"))
                .setMessage(localizedText("「" + name + "」を削除しますか？この操作は取り消せません。", "Delete \"" + name + "\"? This cannot be undone."))
                .setPositiveButton(localizedText("削除", "Delete"), (d, w) -> deleteSelectedConfiguration())
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
        });
        backupProfilesButton.setOnClickListener(v -> performBackup());
        restoreProfilesButton.setOnClickListener(v -> performRestore());
        loadModelButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            startModelAction(true, null);
        });
        selectModelButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            showDownloadedModelPicker();
        });
        importModelButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            launchLocalGgufPicker();
        });
        searchGgufButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            showHuggingFaceSearchDialog();
        });
        maintainModelButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            showModelMaintenanceDialog();
        });
        selectProjectorButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            showStoredProjectorSelectionDialog();
        });
        clearProjectorButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            // Explicit user intent to disable vision: suppress mmproj auto-discovery too.
            selectedProjectorDisabled = true;
            setSelectedProjectorReference("", false);
        });
        applyLoraButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            launchGgufPicker(REQUEST_IMPORT_LORA_ADAPTER, buildDefaultImportUri());
        });
        clearLoraButton.setOnClickListener(v -> {
            if (isBusyActionBlocked()) {
                return;
            }
            clearLoraAdapter();
        });
        // ---- MTP (experimental) controls. The values live in the per-model config
        //      (updateUIFromConfig loads them, collectConfiguration saves them); here we only
        //      wire the listeners. Takes effect on the next model (re)load / config save. ----
        if (mtpEnableToggle != null) {
            mtpEnableToggle.setOnCheckedChangeListener((btn, checked) -> updateMtpControlsEnabled());
        }
        if (mtpModelButton != null) {
            mtpModelButton.setOnClickListener(v -> {
                if (isBusyActionBlocked()) {
                    return;
                }
                showMtpSelectionDialog();
            });
        }
        updateMtpControlsEnabled();
        modelUrlInput.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
            }

            @Override
            public void afterTextChanged(Editable s) {
                clearIncompatibleProjectorSelection();
                updateMultimodalProjectorInfo();
            }
        });
        backButton.setOnClickListener(v -> finish());
        cancelButton.setOnClickListener(v -> cancelAndReturn());
        licenseButton.setOnClickListener(v -> showLicenseDialog());
        documentsButton.setOnClickListener(v -> openDocuments());
        setupApiUrlCopyTarget(apiLoopbackUrlView, localizedText("ローカルURL", "Local URL"));
        setupApiUrlCopyTarget(apiWifiUrlView, localizedText("LAN URL", "LAN URL"));
        setupCollapsibleSections();
        applyLocalizedUiText();
        updateApiServerUrlViews();
        updateActionButtonStateForBusy();
    }

    /** GPU トグルに応じてオフロード Layers スライダーの有効・無効を切り替える。
     *  オフロード (n_gpu_layers) は GPU 有効時のみ適用される (OFF = CPU はスライダー無効)。 */
    private void updateBackendDependentUi() {
        final boolean offloadActive = (gpuEnabledSwitch != null) && gpuEnabledSwitch.isChecked();

        if (gpuLayersSeekBar != null) gpuLayersSeekBar.setEnabled(offloadActive);
        if (gpuLayersValue   != null) gpuLayersValue.setEnabled(offloadActive);
    }

    private void openDocuments() {
        Intent intent = new Intent(this, DocumentsActivity.class);
        startActivity(intent);
    }

    private boolean isBusyActionBlocked() {
        if (importInProgress) {
            updateActionButtonStateForBusy();
            showToast(localizedText("モデル取込を実行中です", "Model import is already running"));
            return true;
        }
        if (huggingFaceSearchInProgress) {
            updateActionButtonStateForBusy();
            showToast(localizedText("Hugging Face の検索を実行中です", "Hugging Face search is already running"));
            return true;
        }
        if (modelManager != null && modelManager.isBusy()) {
            updateActionButtonStateForBusy();
            showToast(localizedText("他のリクエストを処理中です", "Model is busy processing another request"));
            return true;
        }
        return false;
    }

    private void updateActionButtonStateForBusy() {
        boolean isBusy = importInProgress
                || huggingFaceSearchInProgress
                || (modelManager != null && modelManager.isBusy());

        if (saveConfigButton != null) saveConfigButton.setEnabled(!isBusy);
        if (loadConfigButton != null) loadConfigButton.setEnabled(!isBusy);
        if (deleteConfigButton != null) deleteConfigButton.setEnabled(!isBusy);
        if (backupProfilesButton != null) backupProfilesButton.setEnabled(!isBusy);
        if (restoreProfilesButton != null) restoreProfilesButton.setEnabled(!isBusy);
        if (importModelButton != null) importModelButton.setEnabled(!isBusy);
        if (searchGgufButton != null) searchGgufButton.setEnabled(!isBusy);
        if (loadModelButton != null) loadModelButton.setEnabled(!isBusy);
        if (selectModelButton != null) {
            selectModelButton.setEnabled(!isBusy && getDownloadedModelFiles().length > 0);
        }
        if (maintainModelButton != null) maintainModelButton.setEnabled(!isBusy);
        if (selectProjectorButton != null) selectProjectorButton.setEnabled(!isBusy);
        if (clearProjectorButton != null) {
            clearProjectorButton.setEnabled(!isBusy && !selectedProjectorReference.isEmpty());
        }

        if (backButton != null) backButton.setEnabled(true);
        if (cancelButton != null) cancelButton.setEnabled(true);
        if (licenseButton != null) licenseButton.setEnabled(true);
        if (documentsButton != null) documentsButton.setEnabled(true);
        updateMtpControlsEnabled();
    }

    // MTP model picker + n_draft are usable only when the MTP toggle is on and nothing is
    // running; the toggle itself is disabled while busy (like the other action buttons).
    private void updateMtpControlsEnabled() {
        boolean busy = importInProgress || huggingFaceSearchInProgress
                || (modelManager != null && modelManager.isBusy());
        boolean on = mtpEnableToggle != null && mtpEnableToggle.isChecked();
        if (mtpEnableToggle != null) mtpEnableToggle.setEnabled(!busy);
        if (mtpModelButton != null) mtpModelButton.setEnabled(on && !busy);
        if (mtpNDraftInput != null) mtpNDraftInput.setEnabled(on && !busy);
    }

    private int parseMtpNDraft() {
        int n = 2;
        if (mtpNDraftInput != null) {
            try { n = Integer.parseInt(mtpNDraftInput.getText().toString().trim()); }
            catch (NumberFormatException ignore) {}
        }
        if (n < 1) n = 1;
        if (n > 16) n = 16;
        return n;
    }

    private String localizedText(String ja, String en) {
        return Translations.get(this, ja, en);
    }

    private void applyLocalizedUiText() {
        setTitle(localizedText("設定", "Settings"));
        View root = findViewById(android.R.id.content);
        if (root != null) {
            localizeViewTree(root);
        }
    }

    private void localizeViewTree(View view) {
        if (view instanceof TextView) {
            TextView textView = (TextView) view;
            CharSequence currentText = textView.getText();
            if (currentText != null) {
                String translatedText = translateSettingsText(currentText.toString());
                if (!translatedText.equals(currentText.toString())) {
                    textView.setText(translatedText);
                }
            }
            CharSequence currentHint = textView.getHint();
            if (currentHint != null) {
                String translatedHint = translateSettingsHint(currentHint.toString());
                if (!translatedHint.equals(currentHint.toString())) {
                    textView.setHint(translatedHint);
                }
            }
        }

        if (view instanceof ViewGroup) {
            ViewGroup group = (ViewGroup) view;
            for (int i = 0; i < group.getChildCount(); i++) {
                localizeViewTree(group.getChildAt(i));
            }
        }
    }

    private String translateSettingsHint(String hint) {
        if (hint.startsWith("Default: ")) {
            return localizedText("既定値: ", "Default: ") + hint.substring("Default: ".length());
        }
        return Translations.get(this, settingsHintJa(hint), hint);
    }

    private String settingsHintJa(String hint) {
        switch (hint) {
            case "Enter configuration name": return "設定名を入力";
            case "https://... or filename.gguf": return "https://... または filename.gguf";
            case "Enter system prompt (optional)": return "システムプロンプトを入力（任意）";
            case "Enter custom chat template (optional)": return "カスタムチャットテンプレートを入力（任意）";
            default: return hint;
        }
    }

    private String translateSettingsText(String text) {
        return Translations.get(this, settingsTextJa(text), text);
    }

    private String settingsTextJa(String text) {
        switch (text) {
            case "Configuration Management": return "設定管理";
            case "Display Language": return "表示言語";
            case "Profile:": return "プロファイル:";
            case "Configuration Name:": return "設定名:";
            case "Save Config": return "設定を保存";
            case "Delete Config": return "設定を削除";
            case "Load Configuration:": return "設定を読み込み:";
            case "Load Selected Config": return "選択した設定を読み込む";
            case "Backup &amp; Restore": return "バックアップ・復元";
            case "Backup Profiles": return "プロファイルをバックアップ";
            case "Restore Profiles": return "プロファイルを復元";
            case "Model Loading": return "モデル読込み";
            case "Model Management": return "モデル管理";
            case "MTP Settings": return "MTP設定";
            case "Model URL / Imported File:": return "モデルURL / 取込済みファイル:";
            case "Multimodal Projector (mmproj):": return "マルチモーダル Projector (mmproj):";
            case "No multimodal projector selected": return "mmproj は未選択です";
            case "Select mmproj": return "mmproj を選択";
            case "Clear mmproj": return "mmproj を解除";
            case "Search GGUF on Hugging Face": return "Hugging FaceでGGUFを検索";
            case "gguf import from local device": return "ローカル端末からggufを取り込む";
            case "Load Model": return "モデルを読み込む";
            case "Select downloaded model": return "ダウンロード済みモデルを選択";
            case "Get models": return "モデル取得";
            case "Rename / delete models": return "モデルの名称変更・削除";
            case "MAINTAIN MODEL": return "モデル管理";
            case "Model file: (none)": return "モデルファイル: （なし）";
            case "Model Parameters": return "モデルパラメータ";
            case "Context Size (n_ctx):": return "コンテキストサイズ (n_ctx):";
            case "Threads (n_threads):": return "スレッド数 (n_threads):";
            case "Batch Size (n_batch):": return "バッチサイズ (n_batch):";
            case "Max Output Tokens (n_predict, -1 = unlimited):": return "最大出力トークン数 (n_predict, -1=無制限):";
            case "KV Cache Quantization:": return "KVキャッシュ量子化:";
            case "GPU Switch Stabilization:": return "GPU切替安定化:";
            case "Compute Backend (off = CPU):": return "計算バックエンド (OFF = CPU):";
            case "GPU (OpenCL/Adreno) Enabled:": return "GPU (OpenCL/Adreno) 有効:";
            case "Offload Layers (GPU):": return "オフロード層 (GPU):";
            case "Temperature (temp):": return "温度 (temp):";
            case "Top-p (top_p):": return "Top-p (top_p):";
            case "Top-k (top_k):": return "Top-k (top_k):";
            case "Penalty Parameters": return "ペナルティ設定";
            case "Penalty Last N:": return "ペナルティ対象直近N:";
            case "Penalty Repeat:": return "反復ペナルティ:";
            case "Penalty Frequency:": return "頻度ペナルティ:";
            case "Penalty Presence:": return "出現ペナルティ:";
            case "Mirostat Parameters": return "Mirostat 設定";
            case "Mirostat (0=disabled, 1=v1, 2=v2):": return "Mirostat (0=無効, 1=v1, 2=v2):";
            case "Mirostat Tau:": return "Mirostat タウ:";
            case "Mirostat Eta:": return "Mirostat イータ:";
            case "Additional Sampling Parameters": return "追加サンプリング設定";
            case "Min-p:": return "Min-p:";
            case "Typical P:": return "Typical P:";
            case "XTC Probability:": return "XTC 確率:";
            case "XTC Threshold:": return "XTC しきい値:";
            case "Top-N-Sigma:": return "Top-N-Sigma:";
            case "Dynamic Temperature Range:": return "動的温度レンジ:";
            case "Dynamic Temperature Exponent:": return "動的温度指数:";
            case "DRY (Don't Repeat Yourself) Parameters": return "DRY (重複抑制) 設定";
            case "DRY Multiplier:": return "DRY 乗数:";
            case "DRY Base:": return "DRY 基底:";
            case "DRY Allowed Length:": return "DRY 許容長:";
            case "DRY Penalty Last N:": return "DRY ペナルティ直近N:";
            case "DRY Sequence Breakers:": return "DRY シーケンス区切り:";
            case "Output Settings": return "出力設定";
            case "Enable Streaming:": return "ストリーミングを有効化:";
            case "Show Performance Metrics:": return "性能指標を表示:";
            case "Prompt Template": return "プロンプトテンプレート";
            case "System Prompt:": return "システムプロンプト:";
            case "Used when API doesn't provide a system message.": return "API が system メッセージを渡さない場合に使用します。";
            case "Enable Think (chat-template-kwargs.enable_thinking):": return "Thinkを有効化 (chat-template-kwargs.enable_thinking):";
            case "Use memory-map (mmap):": return "メモリマップ (mmap) を使う:";
            case "Auto-expand context (n_ctx):": return "コンテキスト自動拡張 (n_ctx):";
            case "Custom Chat Template:": return "カスタムチャットテンプレート:";
            case "Overrides auto-detection. Use {SYSTEM} and {USER} placeholders.": return "自動判定を上書きします。{SYSTEM} と {USER} プレースホルダーを使用します。";
            case "Auto-selected Prompt Template:": return "自動選択されたプロンプトテンプレート:";
            case "Based on custom template or model family detection.": return "カスタムテンプレートまたはモデル種別判定に基づきます。";
            case "(auto-selected template will appear here)": return "（自動選択されたテンプレートがここに表示されます）";
            case "Llama API Server": return "Llama APIサーバー";
            case "Server Port (default: 11434):": return "サーバーポート (既定: 11434):";
            case "Local URL (tap to open / long-press to copy):": return "ローカルURL（タップで起動・長押しでコピー）:";
            case "LAN URL (tap to open / long-press to copy):": return "LAN URL（タップで起動・長押しでコピー）:";
            case "Connect to Wi-Fi to show the LAN URL.": return "Wi-Fi接続時にLAN URLを表示します。";
            case "MCP Settings": return "MCP設定";
            case "Enable MCP outside Web UI:": return "Web UI 以外でMCPを有効化:";
            case "Enable Function Calling outside Web UI:": return "Web UI 以外でFunction Callingを有効化:";
            case "Available only in Web UI when disabled.": return "無効時は Web UI でのみ利用されます。";
            case "MCP Config JSON (shared):": return "MCPコンフィグJSON（共通）:";
            case "Function Definitions JSON (shared):": return "Function Definitions JSON（共通）:";
            case "Log Settings": return "ログ設定";
            case "Log Level:": return "ログレベル:";
            case "Show License": return "ライセンス表示";
            case "Documents": return "ドキュメント";
            case "SAVE & CLOSE": return "保存して閉じる";
            case "CLOSE": return "閉じる";
            default: return text;
        }
    }

    private void setupCollapsibleSections() {
        if (settingsContentContainer == null) {
            return;
        }

        int[] headerIds = new int[]{
                R.id.configurationManagementHeader,
                R.id.modelSelectionHeader,
                R.id.modelManagementHeader,
                R.id.mtpSettingsHeader,
                R.id.modelParametersHeader,
                R.id.penaltyParametersHeader,
                R.id.mirostatParametersHeader,
                R.id.additionalSamplingParametersHeader,
                R.id.dryParametersHeader,
                R.id.outputSettingsHeader,
                R.id.promptTemplateHeader,
                R.id.apiServerHeader,
                R.id.mcpSettingsHeader,
                R.id.logSettingsHeader
        };

        List<View> originalChildren = new ArrayList<>();
        for (int i = 0; i < settingsContentContainer.getChildCount(); i++) {
            originalChildren.add(settingsContentContainer.getChildAt(i));
        }

        int footerStartIndex = findChildIndexById(originalChildren, R.id.licenseButton);
        if (footerStartIndex < 0) {
            footerStartIndex = originalChildren.size();
        }

        int[] headerIndices = new int[headerIds.length];
        for (int i = 0; i < headerIds.length; i++) {
            headerIndices[i] = findChildIndexById(originalChildren, headerIds[i]);
            if (headerIndices[i] < 0) {
                return;
            }
        }

        settingsContentContainer.removeAllViews();
        collapsibleSections.clear();

        for (int i = 0; i < headerIds.length; i++) {
            View headerCandidate = originalChildren.get(headerIndices[i]);
            if (!(headerCandidate instanceof TextView)) {
                continue;
            }
            TextView headerView = (TextView) headerCandidate;

            int contentStart = headerIndices[i] + 1;
            int contentEnd = (i + 1 < headerIds.length) ? headerIndices[i + 1] : footerStartIndex;

            LinearLayout sectionLayout = new LinearLayout(this);
            sectionLayout.setOrientation(LinearLayout.VERTICAL);
            LinearLayout.LayoutParams sectionParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            sectionParams.bottomMargin = dpToPx(10);
            sectionLayout.setLayoutParams(sectionParams);
            // Card look: white rounded background with internal padding.
            sectionLayout.setBackgroundResource(R.drawable.bg_card);
            sectionLayout.setPadding(dpToPx(16), dpToPx(12), dpToPx(16), dpToPx(12));

            LinearLayout headerRow = new LinearLayout(this);
            headerRow.setOrientation(LinearLayout.HORIZONTAL);
            headerRow.setGravity(Gravity.CENTER_VERTICAL);
            headerRow.setClickable(true);
            headerRow.setFocusable(true);
            headerRow.setPadding(0, dpToPx(4), 0, dpToPx(4));
            headerRow.setMinimumHeight(dpToPx(40));

            headerView.setLayoutParams(new LinearLayout.LayoutParams(
                    0,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    1f
            ));

            // Section icon (mockup style): glyph before the title.
            TextView iconView = new TextView(this);
            LinearLayout.LayoutParams iconParams = new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            );
            iconParams.rightMargin = dpToPx(8);
            iconView.setLayoutParams(iconParams);
            iconView.setText(sectionIconGlyph(headerIds[i]));
            iconView.setTextSize(18f);

            // Expand/collapse chevron, now placed on the right edge of the row.
            TextView indicatorView = new TextView(this);
            indicatorView.setLayoutParams(new LinearLayout.LayoutParams(
                    dpToPx(28),
                    ViewGroup.LayoutParams.WRAP_CONTENT
            ));
            indicatorView.setGravity(Gravity.CENTER);
            indicatorView.setTextSize(16f);
            indicatorView.setTypeface(Typeface.DEFAULT_BOLD);

            headerRow.addView(iconView);
            headerRow.addView(headerView);
            headerRow.addView(indicatorView);

            LinearLayout contentLayout = new LinearLayout(this);
            contentLayout.setOrientation(LinearLayout.VERTICAL);
            contentLayout.setLayoutParams(new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT
            ));

            for (int childIndex = contentStart; childIndex < contentEnd; childIndex++) {
                contentLayout.addView(originalChildren.get(childIndex));
            }

            CollapsibleSectionController controller = new CollapsibleSectionController(indicatorView, contentLayout);
            headerRow.setOnClickListener(v -> setSectionExpanded(controller, !controller.expanded));
            setSectionExpanded(controller, !shouldDefaultCollapse(headerIds[i]));

            sectionLayout.addView(headerRow);
            sectionLayout.addView(contentLayout);
            settingsContentContainer.addView(sectionLayout);
            collapsibleSections.add(controller);
        }

        for (int i = footerStartIndex; i < originalChildren.size(); i++) {
            settingsContentContainer.addView(originalChildren.get(i));
        }
    }

    private void setSectionExpanded(CollapsibleSectionController controller, boolean expanded) {
        controller.expanded = expanded;
        controller.contentView.setVisibility(expanded ? View.VISIBLE : View.GONE);
        controller.indicatorView.setText(expanded ? "▼" : "▶");
    }

    private boolean shouldDefaultCollapse(int headerId) {
        return headerId == R.id.modelParametersHeader
                || headerId == R.id.penaltyParametersHeader
                || headerId == R.id.mirostatParametersHeader
                || headerId == R.id.additionalSamplingParametersHeader
                || headerId == R.id.dryParametersHeader
                || headerId == R.id.mtpSettingsHeader
                || headerId == R.id.mcpSettingsHeader;
    }

    // Emoji glyph shown before each collapsible section title (mockup card style).
    private String sectionIconGlyph(int headerId) {
        if (headerId == R.id.configurationManagementHeader) return "⚙";
        if (headerId == R.id.modelSelectionHeader) return "📦";
        if (headerId == R.id.modelManagementHeader) return "🗂";
        if (headerId == R.id.mtpSettingsHeader) return "⚡";
        if (headerId == R.id.modelParametersHeader) return "🎛";
        if (headerId == R.id.penaltyParametersHeader) return "⚖";
        if (headerId == R.id.mirostatParametersHeader) return "🎯";
        if (headerId == R.id.additionalSamplingParametersHeader) return "🎲";
        if (headerId == R.id.dryParametersHeader) return "🔁";
        if (headerId == R.id.outputSettingsHeader) return "📤";
        if (headerId == R.id.promptTemplateHeader) return "📝";
        if (headerId == R.id.apiServerHeader) return "🌐";
        if (headerId == R.id.mcpSettingsHeader) return "🔌";
        if (headerId == R.id.logSettingsHeader) return "📄";
        return "•";
    }

    private int findChildIndexById(List<View> children, int id) {
        for (int i = 0; i < children.size(); i++) {
            if (children.get(i).getId() == id) {
                return i;
            }
        }
        return -1;
    }

    private int dpToPx(int dp) {
        return Math.round(dp * getResources().getDisplayMetrics().density);
    }

    private void setupLanguageSpinner() {
        if (languageSpinner == null) {
            return;
        }
        if (languageLabel != null) {
            languageLabel.setText(localizedText("表示言語", "Display Language"));
        }
        String[] languages = AppLanguageManager.SUPPORTED_LANGUAGE_LABELS;
        ArrayAdapter<String> languageAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, languages);
        languageAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        languageSpinner.setAdapter(languageAdapter);

        String currentLanguage = AppLanguageManager.getOrInitDisplayLanguage(this);
        int currentIndex = AppLanguageManager.indexOf(currentLanguage);
        if (currentIndex < 0) {
            currentIndex = AppLanguageManager.indexOf(AppLanguageManager.LANGUAGE_EN);
        }
        languageSpinner.setSelection(currentIndex, false);

        languageSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, android.view.View view, int position, long id) {
                if (position < 0 || position >= AppLanguageManager.SUPPORTED_LANGUAGES.length) {
                    return;
                }
                String selectedLanguage = AppLanguageManager.SUPPORTED_LANGUAGES[position];
                String existingLanguage = AppLanguageManager.getOrInitDisplayLanguage(SettingsActivity.this);
                if (!selectedLanguage.equals(existingLanguage)) {
                    AppLanguageManager.saveDisplayLanguage(SettingsActivity.this, selectedLanguage);
                    showToast(localizedText("表示言語を変更しました", "Display language updated"));
                    recreate();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // No-op
            }
        });
    }

    private void setupLogLevelSpinner(int savedLogLevel) {
        String[] levels = new String[] { "MAX_DEBUG", "DEBUG", "INFO", "WARN", "ERROR" };
        ArrayAdapter<String> logAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, levels);
        logAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        logLevelSpinner.setAdapter(logAdapter);
        int selection = Math.max(0, Math.min(levels.length - 1, savedLogLevel));
        logLevelSpinner.setSelection(selection);
        modelManager.getLlama().setLogLevel(selection);
        logLevelSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, android.view.View view, int position, long id) {
                int level = position;
                modelManager.getLlama().setLogLevel(level);
                SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
                prefs.edit().putInt(PREF_LOG_LEVEL, level).apply();
                showToast(localizedText("ログレベルを設定: ", "Log level set to ") + levels[position]);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // No-op
            }
        });
    }

    private void showLicenseDialog() {
        TextView textView = new TextView(this);
        textView.setText(getLicenseText());
        textView.setTextIsSelectable(true);
        textView.setPadding(32, 24, 32, 24);
        textView.setTextSize(12);

        android.widget.ScrollView scrollView = new android.widget.ScrollView(this);
        scrollView.addView(textView);

        new AlertDialog.Builder(this)
            .setTitle(localizedText("ライセンスと第三者通知", "License & Third-Party Notices"))
            .setView(scrollView)
            .setPositiveButton(localizedText("閉じる", "Close"), null)
            .show();
    }

    private String getLicenseText() {
        return "LLM AI Server with llama.cpp\n"
            + "Built with Llama (llama.cpp)\n\n"
            + "Copyright (c) 2026 Mitsuo Kuroda\n\n"
            + "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n\n"
            + "1. Grant of Rights\n"
            + "Subject to the terms and conditions of this License, the Author hereby grants to you a worldwide, royalty-free, non-exclusive license to reproduce, modify, and distribute this software in source or binary form.\n\n"
            + "2. Commercial Use Restriction\n"
            + "Commercial use of this software is strictly prohibited without prior written permission from the Author. \"Commercial use\" includes, but is not limited to, selling the software, using the software to provide paid services, or integrating the software into a commercial product. For commercial inquiries, please contact: micklab2026@gmail.com\n\n"
            + "3. Attribution Requirement\n"
            + "The above copyright notice and this permission notice (including the commercial use restriction) shall be included in all copies or substantial portions of the Software.\n\n"
            + "4. Disclaimer of Warranty\n"
            + "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"
            + "================================================================\n"
            + "THIRD-PARTY SOFTWARE NOTICES\n"
            + "This application includes the following third-party open-source software.\n"
            + "================================================================\n\n"
            + "--- llama.cpp / ggml ---\n"
            + "MIT License\n\n"
            + "Copyright (c) 2023-2026 The ggml authors\n\n"
            + "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\n"
            + "The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\n"
            + "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"
            + "--- nlohmann/json ---\n"
            + "MIT License — Copyright (c) 2013-2025 Niels Lohmann\n"
            + "Licensed under the MIT License (same terms as shown above for llama.cpp/ggml).\n\n"
            + "--- cURL / libcurl ---\n"
            + "Copyright (c) 1996-2024, Daniel Stenberg, daniel@haxx.se, and many contributors. All rights reserved.\n\n"
            + "Permission to use, copy, modify, and distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.\n\n"
            + "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n\n"
            + "--- Mbed TLS ---\n"
            + "Copyright The Mbed TLS Contributors\n"
            + "Licensed under the Apache License, Version 2.0 (see the Apache-2.0 notice below).\n\n"
            + "--- OpenCL Headers (Khronos) ---\n"
            + "Copyright (c) 2008-2026 The Khronos Group Inc.\n"
            + "Licensed under the Apache License, Version 2.0 (see the Apache-2.0 notice below).\n"
            + "Note: the Adreno/OpenCL GPU driver itself is provided by the device's vendor and is NOT included in this application.\n\n"
            + "--- Apache License 2.0 (applies to Mbed TLS and OpenCL Headers above) ---\n"
            + "Licensed under the Apache License, Version 2.0 (the \"License\"); you may not use these files except in compliance with the License. You may obtain a copy of the License at\n\n"
            + "    http://www.apache.org/licenses/LICENSE-2.0\n\n"
            + "Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.\n";
    }
    
    private void loadConfigList() {
        List<String> configs = configManager.listConfigurations();
        // Merged profile field: tapping/focusing shows the full list (no text filtering);
        // typing a new name saves-as. A non-filtering adapter keeps the whole list visible.
        configAdapter = new NoFilterArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, configs);
        configNameInput.setAdapter(configAdapter);
        configNameInput.setOnClickListener(v -> showConfigDropDownSafely());
        configNameInput.setOnFocusChangeListener((v, hasFocus) -> {
            if (hasFocus) {
                showConfigDropDownSafely();
            }
        });
    }

    /**
     * Show the profile dropdown only when it is safe to add a window. When the activity is
     * recreated (e.g. after the process was killed under memory pressure), the framework restores
     * focus to this AutoCompleteTextView inside onRestoreInstanceState — BEFORE the window token
     * exists. Calling showDropDown() then throws WindowManager.BadTokenException on the main
     * thread, which is uncaught and crashes the process in a restart loop ("app can't start").
     * Skip the programmatic-restore case (window not attached / activity going away) and keep a
     * try/catch as a final safety net.
     */
    private void showConfigDropDownSafely() {
        if (configNameInput == null || isFinishing() || isDestroyed()) {
            return;
        }
        if (!configNameInput.isAttachedToWindow() || !configNameInput.hasWindowFocus()) {
            return;
        }
        try {
            configNameInput.showDropDown();
        } catch (android.view.WindowManager.BadTokenException | IllegalStateException e) {
            Log.w(TAG, "Skipped profile dropdown (window not ready)", e);
        }
    }
    
    private void loadConfigurationByName(String name) {
        try {
            currentConfig = configManager.loadConfiguration(name);
            updateUIFromConfig(currentConfig);
            showToast(localizedText("設定を読み込みました: ", "Loaded configuration: ") + name);
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to load configuration: " + name, e);
            showToast(localizedText("設定の読み込みに失敗しました: ", "Failed to load configuration: ") + e.getMessage());
            // Load default
            currentConfig = new ConfigurationManager.Configuration();
            updateUIFromConfig(currentConfig);
        }
    }
    
    private void updateUIFromConfig(ConfigurationManager.Configuration config) {
        configNameInput.setText(config.name, false);
        selectedProjectorReference = normalizeReference(config.multimodalProjectorUrl);
        selectedProjectorManualSelection = config.multimodalProjectorManualSelection;
        selectedProjectorDisabled = config.multimodalProjectorDisabled;
        selectedMtpReference = normalizeReference(config.mtpModelReference);
        if (mtpEnableToggle != null) mtpEnableToggle.setChecked(config.mtpEnabled);
        if (mtpNDraftInput != null) mtpNDraftInput.setText(String.valueOf(config.mtpNDraft > 0 ? config.mtpNDraft : 2));
        updateMtpControlsEnabled();
        modelUrlInput.setText(config.modelUrl);
        nCtxInput.setText(String.valueOf(config.nCtx));
        if (nCtxAutoExpandSwitch != null) {
            nCtxAutoExpandSwitch.setChecked(config.nCtxAutoExpand);
        }
        nThreadsInput.setText(String.valueOf(config.nThreads));
        nBatchInput.setText(String.valueOf(config.nBatch));
        tempInput.setText(String.valueOf(config.temp));
        topPInput.setText(String.valueOf(config.topP));
        topKInput.setText(String.valueOf(config.topK));
        
        // Penalty parameters
        penaltyLastNInput.setText(String.valueOf(config.penaltyLastN));
        penaltyRepeatInput.setText(String.valueOf(config.penaltyRepeat));
        penaltyFreqInput.setText(String.valueOf(config.penaltyFreq));
        penaltyPresentInput.setText(String.valueOf(config.penaltyPresent));
        
        // Mirostat parameters
        mirostatInput.setText(String.valueOf(config.mirostat));
        mirostatTauInput.setText(String.valueOf(config.mirostatTau));
        mirostatEtaInput.setText(String.valueOf(config.mirostatEta));
        
        // Additional sampling parameters
        minPInput.setText(String.valueOf(config.minP));
        typicalPInput.setText(String.valueOf(config.typicalP));
        dynatempRangeInput.setText(String.valueOf(config.dynatempRange));
        dynatempExponentInput.setText(String.valueOf(config.dynatempExponent));
        xtcProbabilityInput.setText(String.valueOf(config.xtcProbability));
        xtcThresholdInput.setText(String.valueOf(config.xtcThreshold));
        topNSigmaInput.setText(String.valueOf(config.topNSigma));
        
        // DRY parameters
        dryMultiplierInput.setText(String.valueOf(config.dryMultiplier));
        dryBaseInput.setText(String.valueOf(config.dryBase));
        dryAllowedLengthInput.setText(String.valueOf(config.dryAllowedLength));
        dryPenaltyLastNInput.setText(String.valueOf(config.dryPenaltyLastN));
        drySequenceBreakersInput.setText(config.drySequenceBreakers);
        
        // Streaming
        streamingSwitch.setChecked(config.streaming);
        // Performance metrics (global pref)
        if (showPerfMetricsSwitch != null) {
            SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
            showPerfMetricsSwitch.setChecked(prefs.getBoolean(PREF_SHOW_PERF_METRICS, false));
        }
        // Max output tokens
        if (nPredictInput != null) {
            nPredictInput.setText(String.valueOf(config.nPredict));
        }

        // KV cache quantization
        if (kvCacheTypeSpinner != null) {
            int selIdx = 0;
            for (int i = 0; i < KV_CACHE_TYPE_IDS.length; i++) {
                if (KV_CACHE_TYPE_IDS[i] == config.kvCacheTypeK) { selIdx = i; break; }
            }
            kvCacheTypeSpinner.setSelection(selIdx);
        }

        int layers = config.gpuOffloadLayers;
        int displayLayers = (layers < 0) ? 40 : layers;
        gpuLayersSeekBar.setProgress(displayLayers);
        gpuLayersValue.setText(displayLayers > 39 ? "ALL" : String.valueOf(displayLayers));
        enableThinkingSwitch.setChecked(config.enableThinking);
        if (useMmapSwitch != null) {
            useMmapSwitch.setChecked(config.useMmap);
        }

        // Compute backend: backendType を GPU トグルへ分解 (GPU=ON, それ以外=OFF=CPU)
        boolean gpuOn = (config.backendType == ConfigurationManager.Configuration.BACKEND_GPU);
        if (gpuEnabledSwitch != null) gpuEnabledSwitch.setChecked(gpuOn);
        updateBackendDependentUi();

        // GPU stabilization
        if (gpuStabSpinner != null) {
            int stab = Math.max(0, Math.min(3, config.gpuSwitchStabilization));
            gpuStabSpinner.setSelection(stab);
        }
        
        // New prompt settings
        systemPromptInput.setText(config.systemPrompt != null ? config.systemPrompt : "");
        customChatTemplateInput.setText(config.customChatTemplate != null ? config.customChatTemplate : "");
        clearIncompatibleProjectorSelection();
        updateMultimodalProjectorInfo();
        updateAutoTemplatePreview(config);
    }

    private void updateAutoTemplatePreview(ConfigurationManager.Configuration config) {
        if (autoSelectedTemplateView == null || config == null) {
            return;
        }
        String ggufChatTemplate = "";
        if (modelManager != null && modelManager.isModelLoaded()) {
            ggufChatTemplate = modelManager.getLlama().getChatTemplate();
        }
        String modelPath = loadedModelPath != null ? loadedModelPath : config.modelUrl;
        String settingsSystemPrompt = config.systemPrompt;
        boolean hasSystem = settingsSystemPrompt != null && !settingsSystemPrompt.isEmpty();
        String systemSource = hasSystem ? "settings" : "none";
        PromptTemplateManager.TemplateSelectionResult selection =
                PromptTemplateManager.selectTemplateWithReason(
                        config.customChatTemplate,
                        ggufChatTemplate,
                        modelPath,
                        hasSystem,
                        systemSource);
        autoSelectedTemplateView.setText(selection.template != null ? selection.template : "");
    }
    
    private ConfigurationManager.Configuration getConfigFromUI() {
        ConfigurationManager.Configuration config = new ConfigurationManager.Configuration();
        
        config.name = configNameInput.getText().toString().trim();
        if (config.name.isEmpty()) {
            config.name = "unnamed";
        }
        
        config.modelUrl = modelUrlInput.getText().toString().trim();
        
        try {
            config.nCtx = Integer.parseInt(nCtxInput.getText().toString());
        } catch (NumberFormatException e) {
            config.nCtx = 2048;
        }
        if (nCtxAutoExpandSwitch != null) {
            config.nCtxAutoExpand = nCtxAutoExpandSwitch.isChecked();
        }

        try {
            config.nThreads = Integer.parseInt(nThreadsInput.getText().toString());
        } catch (NumberFormatException e) {
            config.nThreads = 2;
        }
        
        try {
            config.nBatch = Integer.parseInt(nBatchInput.getText().toString());
        } catch (NumberFormatException e) {
            config.nBatch = ConfigurationManager.Configuration.DEFAULT_N_BATCH;
        }
        
        try {
            config.temp = Double.parseDouble(tempInput.getText().toString());
        } catch (NumberFormatException e) {
            config.temp = 0.7;
        }
        
        try {
            config.topP = Double.parseDouble(topPInput.getText().toString());
        } catch (NumberFormatException e) {
            config.topP = 0.9;
        }
        
        try {
            config.topK = Integer.parseInt(topKInput.getText().toString());
        } catch (NumberFormatException e) {
            config.topK = 40;
        }
        
        if (currentConfig != null && currentConfig.promptTemplate != null && !currentConfig.promptTemplate.isEmpty()) {
            config.promptTemplate = currentConfig.promptTemplate;
        }
        config.multimodalProjectorUrl = normalizeReference(selectedProjectorReference);
        config.multimodalProjectorManualSelection =
                !config.multimodalProjectorUrl.isEmpty() && selectedProjectorManualSelection;
        // Only meaningful when no projector is configured: true = user cleared it, suppress auto-discovery.
        config.multimodalProjectorDisabled =
                config.multimodalProjectorUrl.isEmpty() && selectedProjectorDisabled;
        config.mtpEnabled = mtpEnableToggle != null && mtpEnableToggle.isChecked();
        config.mtpModelReference = normalizeReference(selectedMtpReference);
        config.mtpNDraft = parseMtpNDraft();

        // Penalty parameters
        try {
            config.penaltyLastN = Integer.parseInt(penaltyLastNInput.getText().toString());
        } catch (NumberFormatException e) {
            config.penaltyLastN = 64;
        }
        
        try {
            config.penaltyRepeat = Double.parseDouble(penaltyRepeatInput.getText().toString());
        } catch (NumberFormatException e) {
            config.penaltyRepeat = 1.0;
        }
        
        try {
            config.penaltyFreq = Double.parseDouble(penaltyFreqInput.getText().toString());
        } catch (NumberFormatException e) {
            config.penaltyFreq = 0.0;
        }
        
        try {
            config.penaltyPresent = Double.parseDouble(penaltyPresentInput.getText().toString());
        } catch (NumberFormatException e) {
            config.penaltyPresent = 0.0;
        }
        
        // Mirostat parameters
        try {
            config.mirostat = Integer.parseInt(mirostatInput.getText().toString());
        } catch (NumberFormatException e) {
            config.mirostat = 0;
        }
        
        try {
            config.mirostatTau = Double.parseDouble(mirostatTauInput.getText().toString());
        } catch (NumberFormatException e) {
            config.mirostatTau = 5.0;
        }
        
        try {
            config.mirostatEta = Double.parseDouble(mirostatEtaInput.getText().toString());
        } catch (NumberFormatException e) {
            config.mirostatEta = 0.1;
        }
        
        // Additional sampling parameters
        try {
            config.minP = Double.parseDouble(minPInput.getText().toString());
        } catch (NumberFormatException e) {
            config.minP = 0.05;
        }
        
        try {
            config.typicalP = Double.parseDouble(typicalPInput.getText().toString());
        } catch (NumberFormatException e) {
            config.typicalP = 1.0;
        }
        
        try {
            config.dynatempRange = Double.parseDouble(dynatempRangeInput.getText().toString());
        } catch (NumberFormatException e) {
            config.dynatempRange = 0.0;
        }
        
        try {
            config.dynatempExponent = Double.parseDouble(dynatempExponentInput.getText().toString());
        } catch (NumberFormatException e) {
            config.dynatempExponent = 1.0;
        }
        
        try {
            config.xtcProbability = Double.parseDouble(xtcProbabilityInput.getText().toString());
        } catch (NumberFormatException e) {
            config.xtcProbability = 0.0;
        }
        
        try {
            config.xtcThreshold = Double.parseDouble(xtcThresholdInput.getText().toString());
        } catch (NumberFormatException e) {
            config.xtcThreshold = 0.1;
        }
        
        try {
            config.topNSigma = Double.parseDouble(topNSigmaInput.getText().toString());
        } catch (NumberFormatException e) {
            config.topNSigma = -1.0;
        }
        
        // DRY parameters
        try {
            config.dryMultiplier = Double.parseDouble(dryMultiplierInput.getText().toString());
        } catch (NumberFormatException e) {
            config.dryMultiplier = 0.0;
        }
        
        try {
            config.dryBase = Double.parseDouble(dryBaseInput.getText().toString());
        } catch (NumberFormatException e) {
            config.dryBase = 1.75;
        }
        
        try {
            config.dryAllowedLength = Integer.parseInt(dryAllowedLengthInput.getText().toString());
        } catch (NumberFormatException e) {
            config.dryAllowedLength = 2;
        }
        
        try {
            config.dryPenaltyLastN = Integer.parseInt(dryPenaltyLastNInput.getText().toString());
        } catch (NumberFormatException e) {
            config.dryPenaltyLastN = -1;
        }
        
        config.drySequenceBreakers = drySequenceBreakersInput.getText().toString();
        if (config.drySequenceBreakers.isEmpty()) {
            config.drySequenceBreakers = DEFAULT_DRY_SEQUENCE_BREAKERS;
        }
        
        // Max output tokens
        if (nPredictInput != null) {
            try {
                config.nPredict = Integer.parseInt(nPredictInput.getText().toString());
            } catch (NumberFormatException e) {
                config.nPredict = -1;
            }
        }

        // KV cache quantization
        if (kvCacheTypeSpinner != null) {
            int idx = kvCacheTypeSpinner.getSelectedItemPosition();
            if (idx >= 0 && idx < KV_CACHE_TYPE_IDS.length) {
                config.kvCacheTypeK = KV_CACHE_TYPE_IDS[idx];
                config.kvCacheTypeV = KV_CACHE_TYPE_IDS[idx];
            }
        }

        // Streaming
        config.streaming = streamingSwitch.isChecked();
        int progress = gpuLayersSeekBar.getProgress();
        config.gpuOffloadLayers = (progress > 39) ? -1 : progress;
        config.enableThinking = enableThinkingSwitch.isChecked();
        if (useMmapSwitch != null) {
            config.useMmap = useMmapSwitch.isChecked();
        }

        // Compute backend: GPU トグルから backendType を導出 (OFF = CPU)
        boolean gpuOn = (gpuEnabledSwitch != null) && gpuEnabledSwitch.isChecked();
        config.backendType = gpuOn
                ? ConfigurationManager.Configuration.BACKEND_GPU
                : ConfigurationManager.Configuration.BACKEND_CPU;
        config.npuEnabled = false;

        // GPU stabilization
        if (gpuStabSpinner != null) {
            config.gpuSwitchStabilization = gpuStabSpinner.getSelectedItemPosition();
        }

        // New prompt settings
        config.systemPrompt = systemPromptInput.getText().toString();
        config.customChatTemplate = customChatTemplateInput.getText().toString();
        
        return config;
    }

    private String formatBusyTimeoutLabel(int seconds) {
        if (seconds <= 0) {
            return localizedText("無制限（待ち続ける）", "Unlimited (wait forever)");
        }
        int clamped = Math.min(BUSY_QUEUE_WAIT_MAX_SECONDS, Math.max(BUSY_QUEUE_WAIT_MIN_SECONDS, seconds));
        return clamped + localizedText("秒", "s");
    }

    private int resolveBusyTimeoutFromUi() {
        if (busyTimeoutSeekBar == null) {
            return DEFAULT_BUSY_QUEUE_WAIT_SECONDS;
        }
        int progress = busyTimeoutSeekBar.getProgress();
        if (progress <= 0) {
            return 0; // unlimited
        }
        return Math.min(BUSY_QUEUE_WAIT_MAX_SECONDS, Math.max(BUSY_QUEUE_WAIT_MIN_SECONDS, progress));
    }

    /** Parse and clamp the proactive-recycle interval (minutes) from the UI, falling back to default. */
    private int resolveRecycleIntervalFromUi() {
        if (recoveryRecycleIntervalInput == null) {
            return RecoveryScheduler.DEFAULT_RECYCLE_INTERVAL_MIN;
        }
        int minutes = RecoveryScheduler.DEFAULT_RECYCLE_INTERVAL_MIN;
        try {
            String raw = recoveryRecycleIntervalInput.getText().toString().trim();
            if (!raw.isEmpty()) {
                minutes = Integer.parseInt(raw);
            }
        } catch (NumberFormatException ignored) {
        }
        return Math.max(RecoveryScheduler.MIN_RECYCLE_INTERVAL_MIN,
                Math.min(RecoveryScheduler.MAX_RECYCLE_INTERVAL_MIN, minutes));
    }

    /**
     * Ask the OS to exempt the app from battery optimization so the server keeps serving during Doze.
     * Needed for the keep-awake toggle to actually survive idle sleep. Shows the standard system dialog.
     */
    private void requestIgnoreBatteryOptimizations() {
        String pkg = getPackageName();
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (pm != null && pm.isIgnoringBatteryOptimizations(pkg)) {
            showToast(localizedText("既に電池最適化から除外されています", "Already exempt from battery optimization"));
            return;
        }
        try {
            Intent intent = new Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS);
            intent.setData(Uri.parse("package:" + pkg));
            startActivity(intent);
        } catch (Exception e) {
            // Fallback: open the full battery-optimization list so the user can allowlist manually.
            try {
                startActivity(new Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS));
            } catch (Exception e2) {
                showToast(localizedText("電池最適化設定を開けませんでした", "Could not open battery optimization settings"));
            }
        }
    }

    private int resolveApiPortFromUi() {
        Integer apiPort = parseApiPortFromUi();
        return apiPort != null ? apiPort : OllamaApiServer.DEFAULT_PORT;
    }

    private Integer parseApiPortFromUi() {
        if (apiPortInput == null) {
            return null;
        }
        String rawPort = apiPortInput.getText().toString().trim();
        if (rawPort.isEmpty()) {
            return null;
        }
        int apiPort;
        try {
            apiPort = Integer.parseInt(rawPort);
        } catch (NumberFormatException e) {
            return null;
        }
        return (apiPort >= 1 && apiPort <= 65535) ? apiPort : null;
    }

    private boolean validateApiPortInput() {
        Integer apiPort = parseApiPortFromUi();
        if (apiPort != null) {
            apiPortInput.setError(null);
            return true;
        }

        String message = localizedText(
                "1〜65535 のポート番号を入力してください",
                "Enter a port number between 1 and 65535");
        apiPortInput.setError(message);
        apiPortInput.requestFocus();
        showToast(message);
        return false;
    }

    private boolean saveSharedSettings() {
        if (!validateApiPortInput()) {
            return false;
        }
        String rawMcpConfigJson = mcpConfigJsonInput != null
                ? mcpConfigJsonInput.getText().toString()
                : "";
        if (!McpSettingsHelper.isSharedMcpServersJsonValid(rawMcpConfigJson)) {
            showToast(localizedText(
                    "MCPコンフィグJSONはJSON配列で入力してください",
                    "MCP config JSON must be a JSON array"));
            return false;
        }
        String rawFunctionDefinitionsJson = functionDefinitionsJsonInput != null
                ? functionDefinitionsJsonInput.getText().toString()
                : "";
        if (!McpSettingsHelper.isSharedFunctionDefinitionsJsonValid(rawFunctionDefinitionsJson)) {
            showToast(localizedText(
                    "Function Definitions JSONはJSON配列で入力し、各要素に name を含めてください",
                    "Function Definitions JSON must be a JSON array whose items include a name"));
            return false;
        }

        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        prefs.edit()
                .putInt(PREF_API_PORT, resolveApiPortFromUi())
                .putInt(PREF_BUSY_QUEUE_WAIT_SECONDS, resolveBusyTimeoutFromUi())
                .putBoolean(OllamaForegroundService.PREF_KEEP_AWAKE,
                        keepAwakeSwitch != null && keepAwakeSwitch.isChecked())
                .putBoolean(PREF_SHOW_PERF_METRICS, showPerfMetricsSwitch != null && showPerfMetricsSwitch.isChecked())
                .putBoolean(RecoveryScheduler.PREF_WATCHDOG_ENABLED,
                        recoveryWatchdogSwitch == null || recoveryWatchdogSwitch.isChecked())
                .putBoolean(RecoveryScheduler.PREF_RECYCLE_ENABLED,
                        recoveryRecycleSwitch != null && recoveryRecycleSwitch.isChecked())
                .putInt(RecoveryScheduler.PREF_RECYCLE_INTERVAL_MIN, resolveRecycleIntervalFromUi())
                .apply();
        // Apply the recovery-toggle changes immediately: (re)arm or cancel the alarm chain.
        RecoveryScheduler.ensureScheduled(this);
        // Apply the keep-awake change immediately if the API server is currently enabled.
        if (OllamaForegroundService.isApiEnabled(this)) {
            try {
                Intent powerIntent = new Intent(this, OllamaForegroundService.class);
                powerIntent.setAction(OllamaForegroundService.ACTION_APPLY_POWER);
                startService(powerIntent);
            } catch (Exception ignored) {
            }
        }
        McpSettingsHelper.saveSharedMcpServersJson(this, rawMcpConfigJson);
        McpSettingsHelper.saveSharedFunctionDefinitionsJson(this, rawFunctionDefinitionsJson);
        McpSettingsHelper.saveSharedMcpEnabledOutsideWebUi(
                this,
                sharedMcpEnabledSwitch != null && sharedMcpEnabledSwitch.isChecked());
        McpSettingsHelper.saveSharedFunctionDefinitionsEnabledOutsideWebUi(
                this,
                sharedFunctionCallingEnabledSwitch != null && sharedFunctionCallingEnabledSwitch.isChecked());
        return true;
    }

    private void setupApiUrlCopyTarget(TextView targetView, String label) {
        if (targetView == null) {
            return;
        }
        // Same behavior as the main screen's Web UI URL: tap opens the browser, long-press copies.
        targetView.setOnClickListener(v -> openUrlInBrowser(targetView));
        targetView.setOnLongClickListener(v -> {
            copyUrlToClipboard(targetView, label);
            return true;
        });
    }

    private void openUrlInBrowser(TextView sourceView) {
        if (sourceView == null) {
            return;
        }
        Object tag = sourceView.getTag();
        if (!(tag instanceof String)) {
            return;
        }
        String url = (String) tag;
        if (url.isEmpty()) {
            return;
        }
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
        } catch (Exception e) {
            showToast(localizedText("ブラウザを開けません: ", "Cannot open browser: ") + url);
        }
    }

    private void copyUrlToClipboard(TextView sourceView, String label) {
        if (sourceView == null) {
            return;
        }
        Object tag = sourceView.getTag();
        if (!(tag instanceof String)) {
            return;
        }
        String url = (String) tag;
        if (url.isEmpty()) {
            return;
        }
        ClipboardManager clipboardManager = getSystemService(ClipboardManager.class);
        if (clipboardManager == null) {
            showToast(localizedText("クリップボードを利用できません", "Clipboard is unavailable"));
            return;
        }
        clipboardManager.setPrimaryClip(ClipData.newPlainText(label, url));
        showToast(localizedText("コピーしました: ", "Copied: ") + url);
    }

    private void updateApiServerUrlViews() {
        if (apiLoopbackUrlView == null || apiWifiUrlContainer == null || apiWifiUrlHintView == null) {
            return;
        }

        Integer apiPort = parseApiPortFromUi();
        if (apiPort == null) {
            setCopyableUrl(apiLoopbackUrlView, localizedText(
                    "有効なポート番号を入力するとURLを表示します",
                    "Enter a valid port to show the server URL"), false);
            apiWifiUrlContainer.setVisibility(View.GONE);
            apiWifiUrlHintView.setVisibility(View.VISIBLE);
            apiWifiUrlHintView.setText(localizedText(
                    "有効なポート番号を入力するとLAN URLを表示します",
                    "Enter a valid port to show the LAN URL"));
            return;
        }

        setCopyableUrl(apiLoopbackUrlView, buildServerUrl("127.0.0.1", apiPort), true);
        String wifiIpv4Address = getActiveWifiIpv4Address();
        if (wifiIpv4Address == null || wifiIpv4Address.isEmpty()) {
            apiWifiUrlContainer.setVisibility(View.GONE);
            apiWifiUrlHintView.setVisibility(View.VISIBLE);
            apiWifiUrlHintView.setText(localizedText(
                    "Wi-Fi接続時にLAN URLを表示します",
                    "Connect to Wi-Fi to show the LAN URL."));
            return;
        }

        apiWifiUrlContainer.setVisibility(View.VISIBLE);
        apiWifiUrlHintView.setVisibility(View.GONE);
        setCopyableUrl(apiWifiUrlView, buildServerUrl(wifiIpv4Address, apiPort), true);
    }

    private void setCopyableUrl(TextView targetView, String text, boolean copyEnabled) {
        if (targetView == null) {
            return;
        }
        targetView.setText(text);
        targetView.setTag(copyEnabled ? text : null);
        targetView.setEnabled(copyEnabled);
        targetView.setAlpha(copyEnabled ? 1f : 0.7f);
    }

    private String buildServerUrl(String host, int port) {
        return "http://" + host + ":" + port;
    }

    private String getActiveWifiIpv4Address() {
        if (connectivityManager == null) {
            return null;
        }
        try {
            Network activeNetwork = connectivityManager.getActiveNetwork();
            if (activeNetwork == null) {
                return null;
            }
            NetworkCapabilities capabilities = connectivityManager.getNetworkCapabilities(activeNetwork);
            if (capabilities == null || !capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI)) {
                return null;
            }
            LinkProperties linkProperties = connectivityManager.getLinkProperties(activeNetwork);
            if (linkProperties == null) {
                return null;
            }
            for (LinkAddress linkAddress : linkProperties.getLinkAddresses()) {
                InetAddress address = linkAddress.getAddress();
                if (address instanceof Inet4Address && !address.isLoopbackAddress()) {
                    return address.getHostAddress();
                }
            }
        } catch (SecurityException e) {
            Log.w(TAG, "Unable to inspect Wi-Fi network state", e);
        }
        return null;
    }

    private void registerNetworkCallback() {
        if (connectivityManager == null || networkCallbackRegistered) {
            return;
        }
        try {
            connectivityManager.registerDefaultNetworkCallback(networkCallback);
            networkCallbackRegistered = true;
        } catch (RuntimeException e) {
            Log.w(TAG, "Failed to register network callback", e);
        }
    }

    private void unregisterNetworkCallback() {
        if (connectivityManager == null || !networkCallbackRegistered) {
            return;
        }
        try {
            connectivityManager.unregisterNetworkCallback(networkCallback);
        } catch (RuntimeException e) {
            Log.w(TAG, "Failed to unregister network callback", e);
        } finally {
            networkCallbackRegistered = false;
        }
    }
    
    private void saveCurrentConfiguration() {
        if (!saveSharedSettings()) {
            return;
        }
        ConfigurationManager.Configuration config = getConfigFromUI();
        
        try {
            configManager.saveConfiguration(config);
            currentConfig = config;
            
            // Refresh spinner list
            loadConfigList();
            
            // Reflect the saved profile name in the merged field.
            configNameInput.setText(config.name, false);

            showToast(localizedText("設定を保存しました: ", "Configuration saved: ") + config.name);
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to save configuration", e);
            showToast(localizedText("保存に失敗しました: ", "Failed to save: ") + e.getMessage());
        }
    }
    
    private void loadSelectedConfiguration() {
        String selectedName = configNameInput.getText().toString().trim();
        if (selectedName == null || selectedName.isEmpty()) {
            showToast(localizedText("設定が選択されていません", "No configuration selected"));
            return;
        }
        loadConfigurationByName(selectedName);
    }
    
    private void deleteSelectedConfiguration() {
        String selectedName = configNameInput.getText().toString().trim();
        if (selectedName == null || selectedName.isEmpty()) {
            showToast(localizedText("設定が選択されていません", "No configuration selected"));
            return;
        }
        
        if ("default".equals(selectedName)) {
            showToast(localizedText("デフォルト設定は削除できません", "Cannot delete default configuration"));
            return;
        }
        
        if (configManager.deleteConfiguration(selectedName)) {
            loadConfigList();
            showToast(localizedText("設定を削除しました: ", "Deleted configuration: ") + selectedName);
            // Load default after deletion
            loadConfigurationByName("default");
        } else {
            showToast(localizedText("設定の削除に失敗しました", "Failed to delete configuration"));
        }
    }

    private void showModelMaintenanceDialog() {
        final File[] ggufFiles = getManagedGgufFiles();
        if (ggufFiles.length == 0) {
            showToast(localizedText("ダウンロード済みのモデルが見つかりません", "No downloaded model files found"));
            return;
        }

        String[] fileItems = new String[ggufFiles.length];
        for (int i = 0; i < ggufFiles.length; i++) {
            boolean projector = ModelFileHelper.isLikelyProjectorFilename(ggufFiles[i].getName());
            fileItems[i] = ggufFiles[i].getName()
                    + (projector ? "  [mmproj]" : "")
                    + " (" + ggufFiles[i].length() + " bytes)";
        }

        new AlertDialog.Builder(this)
                .setTitle(localizedText("モデル管理", "Model Maintenance"))
                .setItems(fileItems, (dialog, which) -> showModelFileActionsDialog(ggufFiles[which]))
                .setNegativeButton(localizedText("閉じる", "Close"), null)
                .show();
    }

    private void showModelFileActionsDialog(File file) {
        boolean projector = ModelFileHelper.isLikelyProjectorFilename(file.getName());

        List<String> actions = new ArrayList<>();
        List<Runnable> handlers = new ArrayList<>();
        // "Use as model" only makes sense for a regular model, not an mmproj/projector.
        if (!projector) {
            actions.add(localizedText("このモデルを使用", "Use as model"));
            handlers.add(() -> switchCurrentProfileToDownloadedModel(file));
        }
        actions.add(localizedText("名前を変更", "Rename"));
        handlers.add(() -> promptRenameModelFile(file));
        actions.add(localizedText("削除", "Delete"));
        handlers.add(() -> confirmDeleteModelFile(file));

        new AlertDialog.Builder(this)
                .setTitle(file.getName())
                .setItems(actions.toArray(new String[0]), (dialog, which) -> handlers.get(which).run())
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    private File[] getManagedGgufFiles() {
        List<File> files = new ArrayList<>();
        File modelDir = getModelStorageDir();
        File[] existingFiles = modelDir.listFiles();
        if (existingFiles != null) {
            for (File file : existingFiles) {
                if (file.isFile()
                        && file.length() > 0
                        && ModelFileHelper.isGgufFilename(file.getName())
                        && !file.getName().endsWith(IMPORT_TEMP_SUFFIX)
                        && !PendingModelLoadStore.isMarkerFile(file.getName())
                        && !containsFile(files, file)) {
                    files.add(file);
                }
            }
        }

        if (loadedModelPath != null && !loadedModelPath.isEmpty()) {
            File loadedFile = new File(loadedModelPath);
            if (loadedFile.isFile()
                    && loadedFile.length() > 0
                    && ModelFileHelper.isGgufFilename(loadedFile.getName())
                    && !containsFile(files, loadedFile)) {
                files.add(loadedFile);
            }
        }

        files.sort((a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        return files.toArray(new File[0]);
    }

    private boolean containsFile(List<File> files, File target) {
        for (File file : files) {
            if (file.getAbsolutePath().equals(target.getAbsolutePath())) {
                return true;
            }
        }
        return false;
    }

    private void confirmDeleteModelFile(File modelFile) {
        new AlertDialog.Builder(this)
            .setTitle(localizedText("モデルファイルを削除", "Delete Model File"))
            .setMessage(localizedText("次のモデルファイルを削除しますか？\n\n", "Delete this model file?\n\n") + modelFile.getName())
            .setPositiveButton(localizedText("削除", "Delete"), (dialog, which) -> deleteModelFile(modelFile))
            .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
            .show();
    }

    private void switchCurrentProfileToDownloadedModel(File modelFile) {
        if (modelFile == null || !modelFile.isFile() || modelFile.length() <= 0) {
            showToast(localizedText("選択したモデルファイルを利用できません", "The selected model file is unavailable"));
            return;
        }
        if (importInProgress || modelManager.isBusy()) {
            showToast(localizedText(
                    "モデル処理中はプロファイルを変更できません",
                    "Cannot change the profile while a model operation is running"));
            return;
        }

        modelUrlInput.setText(modelFile.getName());
        ConfigurationManager.Configuration config = getConfigFromUI();
        try {
            configManager.saveConfiguration(config);
            currentConfig = config;
            loadConfigList();

            configNameInput.setText(config.name, false);

            loadedModelPath = null;
            modelLoadedSuccessfully = false;
            modelFileInfo.setText(localizedText(
                    "現在のプロファイルを変更しました: ",
                    "Current profile now uses: ") + modelFile.getName());
            modelProgressBar.setProgress(0);
            lastDownloadProgress = 0;
            updateAutoTemplatePreview(config);
            showToast(localizedText(
                    "現在のプロファイルを更新しました: ",
                    "Updated current profile: ") + config.name);
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to switch current profile to downloaded model", e);
            showToast(localizedText(
                    "プロファイルの更新に失敗しました: ",
                    "Failed to update profile: ") + e.getMessage());
        }
    }

    private void deleteModelFile(File modelFile) {
        if (importInProgress || modelManager.isBusy()) {
            showToast(localizedText("他のリクエストを処理中です", "Model is busy processing another request"));
            return;
        }

        boolean removedLoadedModel = false;
        String currentModelPath = modelManager.getCurrentModelPath();
        if (currentModelPath != null && currentModelPath.equals(modelFile.getAbsolutePath())) {
            modelManager.free();
            removedLoadedModel = true;
        }

        boolean deleted = modelFile.delete();
        if (deleted) {
            if (loadedModelPath != null && loadedModelPath.equals(modelFile.getAbsolutePath())) {
                loadedModelPath = null;
                modelLoadedSuccessfully = false;
                removedLoadedModel = true;
            }
            if (removedLoadedModel) {
                modelFileInfo.setText(localizedText("モデルファイル: （なし）", "Model file: (none)"));
                modelProgressBar.setProgress(0);
                lastDownloadProgress = 0;
            }
            showToast(localizedText("モデルファイルを削除しました: ", "Deleted model file: ") + modelFile.getName());
        } else {
            showToast(localizedText("モデルファイルの削除に失敗しました: ", "Failed to delete model file: ") + modelFile.getName());
        }
    }

    private void promptRenameModelFile(File modelFile) {
        if (importInProgress || modelManager.isBusy()) {
            showToast(localizedText(
                    "モデル処理中は名前を変更できません",
                    "Cannot rename while a model operation is running"));
            return;
        }
        String currentModelPath = modelManager.getCurrentModelPath();
        if (currentModelPath != null && currentModelPath.equals(modelFile.getAbsolutePath())) {
            showToast(localizedText(
                    "読み込み中のモデルは名前を変更できません。先に別モデルへ切り替えてください",
                    "Cannot rename the currently loaded model. Switch models first"));
            return;
        }

        final String oldName = modelFile.getName();
        EditText input = new EditText(this);
        input.setSingleLine(true);
        input.setText(oldName);
        // Pre-select the stem (everything before ".gguf") for quick editing.
        int stemEnd = oldName.toLowerCase(Locale.US).endsWith(".gguf")
                ? oldName.length() - 5
                : oldName.length();
        input.setSelection(0, Math.max(0, stemEnd));

        new AlertDialog.Builder(this)
                .setTitle(localizedText("名前を変更", "Rename file"))
                .setView(input)
                .setPositiveButton(localizedText("変更", "Rename"),
                        (dialog, which) -> renameModelFile(modelFile, input.getText().toString().trim()))
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    private void renameModelFile(File modelFile, String newName) {
        if (newName == null || newName.isEmpty()) {
            showToast(localizedText("ファイル名を入力してください", "Enter a file name"));
            return;
        }
        if (newName.contains("/") || newName.contains("\\")) {
            showToast(localizedText("ファイル名にスラッシュは使えません", "File name cannot contain a slash"));
            return;
        }
        if (!ModelFileHelper.isGgufFilename(newName)) {
            showToast(localizedText(".gguf で終わる名前にしてください", "Name must end with .gguf"));
            return;
        }
        final String oldName = modelFile.getName();
        if (newName.equals(oldName)) {
            return;
        }
        File destFile = new File(modelFile.getParentFile(), newName);
        if (destFile.exists()) {
            showToast(localizedText("同名のファイルが既に存在します", "A file with that name already exists"));
            return;
        }
        // Re-check busy/loaded state right before mutating the file.
        if (importInProgress || modelManager.isBusy()) {
            showToast(localizedText(
                    "モデル処理中は名前を変更できません",
                    "Cannot rename while a model operation is running"));
            return;
        }
        String currentModelPath = modelManager.getCurrentModelPath();
        if (currentModelPath != null && currentModelPath.equals(modelFile.getAbsolutePath())) {
            showToast(localizedText(
                    "読み込み中のモデルは名前を変更できません",
                    "Cannot rename the currently loaded model"));
            return;
        }

        if (!modelFile.renameTo(destFile)) {
            showToast(localizedText("名前の変更に失敗しました: ", "Failed to rename: ") + oldName);
            return;
        }

        // Keep saved profiles and the editing UI pointing at the new local filename so the
        // rename does not trigger a re-download (request #4).
        int updatedProfiles = updateConfigurationReferencesForRename(oldName, newName);
        updateUiReferencesForRename(oldName, newName);

        String suffix = updatedProfiles > 0
                ? localizedText("（更新したプロファイル: ", " (updated profiles: ") + updatedProfiles
                        + localizedText("）", ")")
                : "";
        showToast(localizedText("名前を変更しました: ", "Renamed to: ") + newName + suffix);
    }

    private int updateConfigurationReferencesForRename(String oldName, String newName) {
        int updated = 0;
        for (String configName : configManager.listConfigurations()) {
            try {
                ConfigurationManager.Configuration config = configManager.loadConfiguration(configName);
                boolean changed = false;
                if (oldName.equals(extractFilenameFromUrl(config.modelUrl))) {
                    config.modelUrl = newName;
                    changed = true;
                }
                if (oldName.equals(extractFilenameFromUrl(config.multimodalProjectorUrl))) {
                    config.multimodalProjectorUrl = newName;
                    changed = true;
                }
                if (changed) {
                    configManager.saveConfiguration(config);
                    updated++;
                }
            } catch (IOException | JSONException e) {
                Log.w(TAG, "Failed to update references for renamed model in config: " + configName, e);
            }
        }
        return updated;
    }

    private void updateUiReferencesForRename(String oldName, String newName) {
        if (modelUrlInput != null
                && oldName.equals(extractFilenameFromUrl(modelUrlInput.getText().toString()))) {
            modelUrlInput.setText(newName);
        }
        if (oldName.equals(extractFilenameFromUrl(normalizeReference(selectedProjectorReference)))) {
            setSelectedProjectorReference(newName, selectedProjectorManualSelection);
        }
        currentConfig = getConfigFromUI();
        updateAutoTemplatePreview(currentConfig);
    }

    // Model-family pulldown. Labels double as the server search keyword (index 0 = "any" = no keyword).
    private static final String[] HF_FAMILY_KEYWORDS = {
            "", "Qwen", "Gemma", "Llama", "Mistral", "Phi", "DeepSeek", "GLM", "Yi", "Granite", "SmolLM"};
    // Parameter-size pulldown, parallel min/max bounds in billions (0 = unbounded).
    // Parameter-size pulldown bands (parallel min/max in billions; 0 = unbounded), finer-grained.
    private static final double[] HF_SIZE_MIN_B = {0, 0,    0.5, 1,   2, 4,   7, 9,  14, 20, 34, 70};
    private static final double[] HF_SIZE_MAX_B = {0, 0.5,  1,   2,   4, 7,   9, 14, 20, 34, 70, 0};
    // Quantization pulldown (index 0 = "any"). Matched as a substring against GGUF file names.
    // "Q1" matches IQ1_S/IQ1_M 1-bit files (lowercased "iq1_s" contains "q1").
    private static final String[] HF_QUANT_VALUES = {
            "", "Q1", "Q2_K", "Q3_K_M", "Q4_K_M", "Q4_0", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16"};
    // Representative parameter count (in billions) per size band, parallel to buildSizeLabels() /
    // HF_SIZE_MIN_B. 0 = "Any" (no device-suitability rating). Used to estimate the +…+++++ rating.
    private static final double[] HF_SIZE_REP_B = {0, 0.35, 0.75, 1.5, 3, 5.5, 8, 11, 17, 27, 50, 80};

    private void showHuggingFaceSearchDialog() {
        int pad = Math.round(getResources().getDisplayMetrics().density * 20f);

        LinearLayout form = new LinearLayout(this);
        form.setOrientation(LinearLayout.VERTICAL);
        form.setPadding(pad, pad, pad, pad);

        EditText queryInput = new EditText(this);
        queryInput.setSingleLine(true);
        queryInput.setHint(localizedText("モデル名またはキーワード（自由入力・任意）",
                "Model name or keyword (free text, optional)"));
        form.addView(queryInput);

        // ---- Model name: a pulldown (same form as size/quant) + a one-line strength note below.
        Spinner familySpinner = addLabeledSpinner(form,
                localizedText("モデル名", "Model name"),
                buildFamilyLabels());
        TextView familyNote = addNoteView(form);
        final String[] familyStrengths = buildFamilyStrengths();
        familySpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                familyNote.setText(position >= 0 && position < familyStrengths.length
                        ? familyStrengths[position] : "");
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) { }
        });

        // ---- Parameter size: pulldown + a device-specific suitability rating (+ … +++++) below.
        Spinner sizeSpinner = addLabeledSpinner(form,
                localizedText("パラメータ数", "Parameter size"),
                buildSizeLabels());
        TextView sizeRating = addNoteView(form);
        sizeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                sizeRating.setText(paramRatingText(position));
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) { }
        });

        // ---- Quantization: pulldown + a one-line tendency note below.
        Spinner quantSpinner = addLabeledSpinner(form,
                localizedText("量子化", "Quantization"),
                buildQuantLabels());
        TextView quantNote = addNoteView(form);
        final String[] quantTendencies = buildQuantTendencies();
        quantSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                quantNote.setText(position >= 0 && position < quantTendencies.length
                        ? quantTendencies[position] : "");
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) { }
        });

        CheckBox multimodalCheck = new CheckBox(this);
        multimodalCheck.setText(localizedText("マルチモーダル対応のみ", "Multimodal only"));
        form.addView(multimodalCheck);

        CheckBox mtpCheck = new CheckBox(this);
        mtpCheck.setText(localizedText("MTP対応のみ（推定）", "MTP only (best-effort)"));
        form.addView(mtpCheck);

        ScrollView scroll = new ScrollView(this);
        scroll.addView(form);

        new AlertDialog.Builder(this)
                .setTitle(localizedText("Hugging Face で GGUF を検索", "Search GGUF on Hugging Face"))
                .setView(scroll)
                .setPositiveButton(localizedText("検索", "Search"),
                        (dialog, which) -> {
                            int familyIdx = familySpinner.getSelectedItemPosition();
                            int sizeIdx = sizeSpinner.getSelectedItemPosition();
                            int quantIdx = quantSpinner.getSelectedItemPosition();
                            HuggingFaceApiClient.SearchFilters filters =
                                    new HuggingFaceApiClient.SearchFilters(
                                            queryInput.getText().toString().trim(),
                                            HF_FAMILY_KEYWORDS[familyIdx],
                                            HF_SIZE_MIN_B[sizeIdx],
                                            HF_SIZE_MAX_B[sizeIdx],
                                            multimodalCheck.isChecked(),
                                            mtpCheck.isChecked());
                            huggingFaceQuantFilter = HF_QUANT_VALUES[quantIdx];
                            searchHuggingFaceRepositories(filters);
                        })
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    /** Adds a small secondary-text note line (used under the model / size / quant pulldowns). */
    private TextView addNoteView(LinearLayout parent) {
        TextView note = new TextView(this);
        note.setTextSize(12f);
        note.setTextColor(0xFF888888);
        int topPad = Math.round(getResources().getDisplayMetrics().density * 2f);
        note.setPadding(0, topPad, 0, 0);
        parent.addView(note, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return note;
    }

    /** One-line strengths parallel to HF_FAMILY_KEYWORDS / buildFamilyLabels(). */
    private String[] buildFamilyStrengths() {
        return new String[]{
                localizedText("すべてのモデルを対象", "Search across all models"),
                localizedText("多言語・コーディング・長文に強い万能型", "Versatile: multilingual, coding, long context"),
                localizedText("軽量で多言語・要約が得意", "Lightweight; strong multilingual & summarization"),
                localizedText("汎用的で派生・エコシステムが豊富", "General-purpose with a rich ecosystem"),
                localizedText("少パラメータで高効率・指示追従が堅実", "Efficient at small sizes; solid instruction following"),
                localizedText("小型ながら推論・数学に強い", "Small but strong at reasoning & math"),
                localizedText("推論・コード生成・数学に特化", "Specialized in reasoning, code & math"),
                localizedText("中英バイリンガルとツール利用に強い", "Strong CN/EN bilingual & tool use"),
                localizedText("長文コンテキストとバイリンガル性能", "Long context & bilingual performance"),
                localizedText("企業向け・コード/RAG志向で安定", "Enterprise-oriented; stable for code/RAG"),
                localizedText("超小型・端末常駐向けの最軽量", "Ultra-small; ideal for always-on/on-device")};
    }

    /** One-line quantization tendencies parallel to HF_QUANT_VALUES / buildQuantLabels(). */
    private String[] buildQuantTendencies() {
        return new String[]{
                localizedText("量子化を指定しません", "No quantization filter"),
                localizedText("1bit級。極小・最速だが品質は実験的", "1-bit class; tiny/fastest but experimental quality"),
                localizedText("最小・最速だが品質低下が大きい（動作確認向け）", "Smallest/fastest but large quality loss (smoke test)"),
                localizedText("小型・高速だが品質はやや低下", "Small/fast, slightly reduced quality"),
                localizedText("サイズと品質のバランスが最良（推奨既定）", "Best size/quality balance (recommended default)"),
                localizedText("旧式4bit。軽量だがQ4_K_Mに品質で劣る", "Legacy 4-bit; light but lower quality than Q4_K_M"),
                localizedText("高品質寄りで中容量。バランス良好", "Higher quality, medium size, well balanced"),
                localizedText("ほぼ無損失に近い高品質。容量大", "Near-lossless quality; large size"),
                localizedText("高品質・大容量。速度/メモリ負担大", "High quality, large; heavier on speed/memory"),
                localizedText("非量子化(16bit)。最高品質だが最大容量・低速", "Unquantized (16-bit); top quality, largest & slow"),
                localizedText("非量子化(bfloat16)。F16同等で最大容量", "Unquantized (bfloat16); like F16, largest size")};
    }

    /**
     * Device-specific suitability rating (+ … +++++) for a parameter-size band, based on total RAM.
     * + = hard to run, +++ = practically usable, +++++ = fast. Rough estimate using a Q4_K_M
     * footprint of ~0.7 GB per 1B params against ~half of total device memory.
     */
    private String paramRatingText(int sizeIdx) {
        double repB = (sizeIdx >= 0 && sizeIdx < HF_SIZE_REP_B.length) ? HF_SIZE_REP_B[sizeIdx] : 0;
        if (repB <= 0) {
            return localizedText("この端末での推奨度: 指定なし", "Suitability on this device: not specified");
        }
        long totalBytes = getDeviceTotalMemoryBytes();
        double deviceGB = totalBytes > 0 ? totalBytes / 1.0e9 : 0;
        double requiredGB = repB * 0.7;                 // Q4_K_M weights + modest overhead
        double budgetGB = deviceGB > 0 ? deviceGB * 0.5 : 0;  // usable for model weights
        int stars;
        if (budgetGB <= 0) {
            stars = 3;                                  // unknown device → neutral
        } else {
            double ratio = requiredGB / budgetGB;
            if (ratio <= 0.25) stars = 5;
            else if (ratio <= 0.5) stars = 4;
            else if (ratio <= 0.8) stars = 3;
            else if (ratio <= 1.1) stars = 2;
            else stars = 1;
        }
        String plus = "+++++".substring(0, stars);
        String note;
        switch (stars) {
            case 5:  note = localizedText("高速実行", "fast"); break;
            case 4:  note = localizedText("快適", "comfortable"); break;
            case 3:  note = localizedText("実用可能", "usable"); break;
            case 2:  note = localizedText("動作は重い", "sluggish"); break;
            default: note = localizedText("実行困難", "hard to run"); break;
        }
        return localizedText("この端末での推奨度: ", "Suitability on this device: ") + plus + " (" + note + ")";
    }

    /** Total physical device memory in bytes (ActivityManager.MemoryInfo.totalMem), or -1. */
    private long getDeviceTotalMemoryBytes() {
        try {
            android.app.ActivityManager am =
                    (android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE);
            if (am != null) {
                android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo();
                am.getMemoryInfo(mi);
                return mi.totalMem;
            }
        } catch (Throwable ignored) {
        }
        return -1L;
    }

    /** Adds a caption + full-width {@link Spinner} to {@code parent} and returns the spinner. */
    private Spinner addLabeledSpinner(LinearLayout parent, String caption, String[] labels) {
        int topMargin = Math.round(getResources().getDisplayMetrics().density * 12f);
        TextView label = new TextView(this);
        label.setText(caption);
        LinearLayout.LayoutParams labelParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        labelParams.topMargin = topMargin;
        parent.addView(label, labelParams);

        Spinner spinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, labels);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        parent.addView(spinner, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        return spinner;
    }

    private String[] buildFamilyLabels() {
        String[] labels = HF_FAMILY_KEYWORDS.clone();
        labels[0] = localizedText("指定なし", "Any");
        return labels;
    }

    private String[] buildSizeLabels() {
        return new String[]{
                localizedText("指定なし", "Any"),
                localizedText("〜0.5B", "Up to 0.5B"),
                "0.5–1B",
                "1–2B",
                "2–4B",
                "4–7B",
                "7–9B",
                "9–14B",
                "14–20B",
                "20–34B",
                "34–70B",
                localizedText("70B以上", "70B+")};
    }

    private String[] buildQuantLabels() {
        String[] labels = HF_QUANT_VALUES.clone();
        labels[0] = localizedText("指定なし", "Any");
        return labels;
    }

    private void searchHuggingFaceRepositories(HuggingFaceApiClient.SearchFilters filters) {
        setHuggingFaceSearchBusy(true, localizedText(
                "Hugging Face を検索中... ",
                "Searching Hugging Face... "));

        new Thread(() -> {
            try {
                List<HuggingFaceApiClient.ModelSearchResult> results =
                        HuggingFaceApiClient.searchGgufModels(filters);
                runOnUiThread(() -> {
                    setHuggingFaceSearchBusy(false, null);
                    if (results.isEmpty()) {
                        showToast(localizedText(
                                "該当するGGUFリポジトリが見つかりませんでした",
                                "No matching GGUF repositories were found"));
                        return;
                    }
                    showHuggingFaceRepositoryDialog(results);
                });
            } catch (IOException | JSONException e) {
                Log.e(TAG, "Failed to search Hugging Face GGUF repositories", e);
                runOnUiThread(() -> {
                    setHuggingFaceSearchBusy(false, null);
                    showToast(localizedText(
                            "Hugging Face の検索に失敗しました: ",
                            "Hugging Face search failed: ") + e.getMessage());
                });
            }
        }, "hf-gguf-search").start();
    }

    private void showHuggingFaceRepositoryDialog(List<HuggingFaceApiClient.ModelSearchResult> results) {
        String[] labels = new String[results.size()];
        for (int i = 0; i < results.size(); i++) {
            HuggingFaceApiClient.ModelSearchResult result = results.get(i);
            StringBuilder label = new StringBuilder(result.getRepoId());
            String meta = buildHuggingFaceRepositoryMeta(result);
            if (!meta.isEmpty()) {
                label.append("  (").append(meta).append(')');
            }
            labels[i] = label.toString();
        }

        new AlertDialog.Builder(this)
                .setTitle(localizedText("ダウンロード元を選択", "Select a repository"))
                .setItems(labels, (dialog, which) -> fetchHuggingFaceRepositoryFiles(results.get(which)))
                .setNegativeButton(localizedText("閉じる", "Close"), null)
                .show();
    }

    private void fetchHuggingFaceRepositoryFiles(HuggingFaceApiClient.ModelSearchResult result) {
        setHuggingFaceSearchBusy(true, localizedText(
                "GGUF 一覧を取得中... ",
                "Loading GGUF files... ") + result.getRepoId());

        new Thread(() -> {
            try {
                HuggingFaceApiClient.RepositoryFiles repositoryFiles =
                        HuggingFaceApiClient.getRepositoryGgufFiles(result.getRepoId());
                runOnUiThread(() -> {
                    setHuggingFaceSearchBusy(false, null);
                    if (repositoryFiles.getFiles().isEmpty()) {
                        showToast(localizedText(
                                "ダウンロード可能なGGUFファイルが見つかりませんでした",
                                "No downloadable GGUF files were found"));
                        return;
                    }
                    showHuggingFaceFileDialog(repositoryFiles);
                });
            } catch (IOException | JSONException e) {
                Log.e(TAG, "Failed to load Hugging Face GGUF file list", e);
                runOnUiThread(() -> {
                    setHuggingFaceSearchBusy(false, null);
                    showToast(localizedText(
                            "GGUF 一覧の取得に失敗しました: ",
                            "Failed to load GGUF files: ") + e.getMessage());
                });
            }
        }, "hf-gguf-files").start();
    }

    private void showHuggingFaceFileDialog(HuggingFaceApiClient.RepositoryFiles repositoryFiles) {
        List<HuggingFaceApiClient.GgufFileInfo> allFiles = repositoryFiles.getFiles();

        // Quantization is a per-file property, so the quant chosen in the search dialog is applied
        // here. If nothing matches, fall back to the full list rather than showing an empty dialog.
        List<HuggingFaceApiClient.GgufFileInfo> files = allFiles;
        boolean quantFallback = false;
        if (huggingFaceQuantFilter != null && !huggingFaceQuantFilter.isEmpty()) {
            String needle = huggingFaceQuantFilter.toLowerCase(Locale.US);
            List<HuggingFaceApiClient.GgufFileInfo> filtered = new ArrayList<>();
            for (HuggingFaceApiClient.GgufFileInfo file : allFiles) {
                if (file.getFilename().toLowerCase(Locale.US).contains(needle)) {
                    filtered.add(file);
                }
            }
            if (filtered.isEmpty()) {
                quantFallback = true;
            } else {
                files = filtered;
            }
        }
        if (quantFallback) {
            showToast(localizedText(
                    "指定した量子化のファイルが無いため全件を表示します",
                    "No files for the selected quantization; showing all"));
        }

        // Combine model files (quant-filtered above) with the repository's mmproj/projector files.
        // Projector files are always shown regardless of the quant filter (they are named like
        // "mmproj-…-F16.gguf" and would otherwise be filtered out), tagged with "[mmproj]", and are
        // downloaded directly WITHOUT loading (request: search/download mmproj directly, no load).
        final List<HuggingFaceApiClient.GgufFileInfo> shownFiles = new ArrayList<>(files);
        for (HuggingFaceApiClient.GgufFileInfo projector : repositoryFiles.getProjectorFiles()) {
            if (!shownFiles.contains(projector)) {
                shownFiles.add(projector);
            }
        }
        String[] labels = new String[shownFiles.size()];
        for (int i = 0; i < shownFiles.size(); i++) {
            HuggingFaceApiClient.GgufFileInfo file = shownFiles.get(i);
            labels[i] = file.isProjector()
                    ? "[mmproj] " + file.getFilename()
                    : file.getFilename();
        }

        new AlertDialog.Builder(this)
                .setTitle(localizedText("ダウンロード対象を選択", "Select a GGUF file"))
                .setItems(labels, (dialog, which) -> {
                    HuggingFaceApiClient.GgufFileInfo chosen = shownFiles.get(which);
                    if (chosen.isProjector()) {
                        downloadProjectorFileOnly(chosen);
                    } else {
                        applyHuggingFaceFileSelection(repositoryFiles, chosen);
                    }
                })
                .setNegativeButton(localizedText("閉じる", "Close"), null)
                .show();
    }

    /**
     * Download a standalone mmproj/projector file directly, WITHOUT loading it (request #4). The
     * file is saved into the model storage directory; if the name collides with an existing file a
     * " (N)" suffix is added so nothing is overwritten. No model/projector config is changed.
     */
    private void downloadProjectorFileOnly(HuggingFaceApiClient.GgufFileInfo projector) {
        if (modelManager.isBusy()) {
            showToast(localizedText("他のリクエストを処理中です", "Model is busy processing another request"));
            return;
        }
        final String url = projector.getDownloadUrl();
        modelFileInfo.setText(localizedText("mmproj をダウンロード中: ", "Downloading mmproj: ")
                + projector.getFilename());
        modelProgressBar.setIndeterminate(true);
        new Thread(() -> {
            if (!modelManager.tryAcquire()) {
                runOnUiThread(() -> showToast(localizedText("モデルは処理中です", "Model is busy")));
                return;
            }
            String savedName = null;
            try {
                savedName = modelManager.downloadStandaloneFile(url);
            } catch (Throwable t) {
                Log.e(TAG, "mmproj download error", t);
            } finally {
                modelManager.release();
            }
            final String saved = savedName;
            runOnUiThread(() -> {
                modelProgressBar.setIndeterminate(false);
                if (saved != null) {
                    modelFileInfo.setText(localizedText("mmproj ダウンロード完了: ", "mmproj downloaded: ") + saved);
                    modelProgressBar.setProgress(100);
                    lastDownloadProgress = 100;
                    showToast(localizedText("mmproj をダウンロードしました（ロードはしません）",
                            "Downloaded mmproj (not loaded)"));
                } else {
                    modelFileInfo.setText(localizedText("mmproj のダウンロードに失敗しました", "mmproj download failed"));
                    modelProgressBar.setProgress(0);
                    lastDownloadProgress = 0;
                    showToast(localizedText("mmproj のダウンロードに失敗しました", "mmproj download failed"));
                }
            });
        }, "hf-mmproj-download").start();
    }

    private void applyHuggingFaceFileSelection(
            HuggingFaceApiClient.RepositoryFiles repositoryFiles,
            HuggingFaceApiClient.GgufFileInfo selectedFile) {
        String downloadUrl = selectedFile.getDownloadUrl();
        String previousModelReference = modelUrlInput.getText().toString().trim();
        String previousProjectorReference = selectedProjectorReference;
        boolean previousProjectorManualSelection = selectedProjectorManualSelection;
        boolean previousProjectorDisabled = selectedProjectorDisabled;
        modelUrlInput.setText(downloadUrl);
        HuggingFaceApiClient.GgufFileInfo suggestedProjectorInfo =
                repositoryFiles.findMatchingProjector(selectedFile, false, false);
        if (suggestedProjectorInfo != null) {
            setSelectedProjectorReference(suggestedProjectorInfo.getDownloadUrl(), false);
        } else {
            setSelectedProjectorReference("", false);
        }
        currentConfig = getConfigFromUI();
        updateAutoTemplatePreview(currentConfig);

        modelFileInfo.setText(localizedText(
                "選択したモデル: ",
                "Selected model: ") + selectedFile.getFilename());
        startModelAction(
                true,
                () -> {
                    modelUrlInput.setText(previousModelReference);
                    setSelectedProjectorReference(previousProjectorReference, previousProjectorManualSelection);
                    selectedProjectorDisabled = previousProjectorDisabled;
                    currentConfig = getConfigFromUI();
                    updateAutoTemplatePreview(currentConfig);
                });
    }

    private void setHuggingFaceSearchBusy(boolean busy, String statusMessage) {
        huggingFaceSearchInProgress = busy;
        if (busy) {
            modelProgressBar.setIndeterminate(true);
            modelProgressBar.setProgress(0);
            lastDownloadProgress = 0;
            if (statusMessage != null && !statusMessage.isEmpty()) {
                modelFileInfo.setText(statusMessage);
            }
        } else {
            modelProgressBar.setIndeterminate(false);
            if (modelProgressBar.getProgress() == 0) {
                lastDownloadProgress = 0;
            }
        }
        updateActionButtonStateForBusy();
    }

    private String buildHuggingFaceRepositoryMeta(HuggingFaceApiClient.ModelSearchResult result) {
        List<String> items = new ArrayList<>();
        if (result.getParamSizeB() > 0d) {
            items.add(formatParamSize(result.getParamSizeB()));
        }
        if (result.isMultimodal()) {
            items.add(localizedText("マルチモーダル", "Multimodal"));
        }
        if (result.getDownloads() > 0) {
            items.add("DL " + formatCompactCount(result.getDownloads()));
        }
        if (result.getLikes() > 0) {
            items.add(localizedText("いいね ", "Likes ") + formatCompactCount(result.getLikes()));
        }
        if (result.getPipelineTag() != null && !result.getPipelineTag().isEmpty()) {
            items.add(result.getPipelineTag());
        }
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) {
                builder.append(" / ");
            }
            builder.append(items.get(i));
        }
        return builder.toString();
    }

    private String formatParamSize(double paramSizeB) {
        if (paramSizeB == Math.floor(paramSizeB)) {
            return String.format(Locale.US, "%.0fB", paramSizeB);
        }
        return String.format(Locale.US, "%.1fB", paramSizeB);
    }

    private String formatCompactCount(long value) {
        if (value >= 1_000_000_000L) {
            return String.format(Locale.US, "%.1fB", value / 1_000_000_000.0d);
        }
        if (value >= 1_000_000L) {
            return String.format(Locale.US, "%.1fM", value / 1_000_000.0d);
        }
        if (value >= 1_000L) {
            return String.format(Locale.US, "%.1fK", value / 1_000.0d);
        }
        return String.valueOf(value);
    }

    private void launchLocalGgufPicker() {
        launchGgufPicker(REQUEST_IMPORT_MODEL_LOCAL_DEVICE, buildDefaultImportUri());
    }

    private void launchGgufPicker(int requestCode, Uri initialUri) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_LOCAL_ONLY, true);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        if (initialUri != null) {
            intent.putExtra(DocumentsContract.EXTRA_INITIAL_URI, initialUri);
        }
        try {
            startActivityForResult(intent, requestCode);
        } catch (ActivityNotFoundException e) {
            Log.e(TAG, "No document picker available for GGUF import", e);
            showToast(localizedText("ファイル選択画面を開けませんでした", "Could not open the file picker"));
        } catch (IllegalArgumentException e) {
            Log.e(TAG, "Failed to launch GGUF picker", e);
            showToast(localizedText("ファイル選択画面を開けませんでした", "Could not open the file picker"));
        }
    }

    private Uri buildDefaultImportUri() {
        String documentId = "primary:Download";
        try {
            return DocumentsContract.buildDocumentUri("com.android.externalstorage.documents", documentId);
        } catch (IllegalArgumentException e) {
            Log.w(TAG, "Failed to build initial picker URI for " + documentId, e);
            return null;
        }
    }

    private void importModelFromUri(Uri sourceUri) {
        ImportedModelCandidate candidate = resolveImportedModelCandidate(sourceUri);
        String displayName = candidate.displayName;
        if (displayName == null || displayName.isEmpty()) {
            showToast(localizedText("選択したファイル名を取得できません", "Could not determine the selected file name"));
            return;
        }
        if (!ModelFileHelper.isGgufFilename(displayName)) {
            showToast(localizedText(".gguf ファイルを選択してください", "Please select a .gguf file"));
            return;
        }

        File destFile = new File(getModelStorageDir(), displayName);
        String currentModelPath = modelManager.getCurrentModelPath();
        if (currentModelPath != null && currentModelPath.equals(destFile.getAbsolutePath())) {
            showToast(localizedText(
                    "読み込み中のモデルは上書きできません。先に別モデルへ切り替えるか削除してください",
                    "The currently loaded model cannot be replaced. Switch models or delete it first"));
            return;
        }

        importInProgress = true;
        updateActionButtonStateForBusy();
        modelProgressBar.setIndeterminate(candidate.sizeBytes <= 0);
        modelProgressBar.setProgress(0);
        lastDownloadProgress = 0;
        modelFileInfo.setText(localizedText("モデルを取り込み中... ", "Importing model... ") + displayName);

        new Thread(() -> {
            String error = copyImportedModelToStorage(sourceUri, destFile, candidate.sizeBytes, displayName);
            runOnUiThread(() -> {
                importInProgress = false;
                modelProgressBar.setIndeterminate(false);
                updateActionButtonStateForBusy();

                if (error != null) {
                    modelProgressBar.setProgress(0);
                    modelFileInfo.setText(localizedText("モデル取込に失敗しました", "Model import failed"));
                    showToast(error);
                    return;
                }

                loadedModelPath = null;
                modelLoadedSuccessfully = false;
                boolean importedProjector = ModelFileHelper.isLikelyProjectorFilename(displayName);
                if (importedProjector) {
                    modelFileInfo.setText(localizedText("mmproj を取り込みました: ", "Imported mmproj: ")
                            + displayName + " (" + destFile.length() + " bytes)");
                    String modelReference = modelUrlInput.getText().toString().trim();
                    if (!modelReference.isEmpty()
                            && ModelFileHelper.canAutoApplyProjectorReference(modelReference, displayName)) {
                        setSelectedProjectorReference(displayName, false);
                    }
                } else {
                    modelUrlInput.setText(displayName);
                    modelFileInfo.setText(localizedText("モデルファイル: ", "Model file: ") + displayName + " (" + destFile.length() + " bytes)");
                }
                modelProgressBar.setProgress(0);
                currentConfig = getConfigFromUI();
                showToast(importedProjector
                        ? localizedText("mmproj を取り込みました: ", "Imported mmproj: ") + displayName
                        : localizedText("モデルファイルを取り込みました: ", "Imported model file: ") + displayName);
                updateAutoTemplatePreview(currentConfig);
            });
        }).start();
    }

    private void showStoredProjectorSelectionDialog() {
        String modelReference = modelUrlInput.getText().toString().trim();
        if (modelReference.isEmpty()) {
            showToast(localizedText("先にモデルを選択してください", "Select a model first"));
            return;
        }

        File[] projectorFiles = getDownloadedProjectorFiles();
        if (projectorFiles.length == 0) {
            showToast(localizedText("利用可能な GGUF ファイルが見つかりません", "No stored GGUF files were found"));
            return;
        }

        String[] labels = new String[projectorFiles.length];
        String currentSelection = extractFilenameFromUrl(normalizeReference(selectedProjectorReference));
        int selectedIndex = 0;
        for (int i = 0; i < projectorFiles.length; i++) {
            File projectorFile = projectorFiles[i];
            boolean recommended = ModelFileHelper.canAutoApplyProjectorReference(modelReference, projectorFile.getName());
            labels[i] = projectorFile.getName()
                    + (recommended ? localizedText("  [推奨]", "  [Recommended]") : "")
                    + " (" + projectorFile.length() + " bytes)";
            if (projectorFile.getName().equalsIgnoreCase(currentSelection)) {
                selectedIndex = i;
            }
        }

        showScrollableSingleChoiceDialog(
                localizedText("mmproj を選択", "Select mmproj"),
                labels,
                selectedIndex,
                localizedText("選択", "Select"),
                index -> selectProjectorWithCompatibilityCheck(modelReference, projectorFiles[index]),
                localizedText("解除", "Clear"),
                index -> setSelectedProjectorReference("", false));
    }

    // Experimental: pick the MTP-head draft GGUF (or disable). Persisted to prefs and
    // applied by ModelManager at the next model load (llama.setSpeculative before init).
    // Picks the MTP draft source (the enable/disable is owned by the toggle). Empty path =
    // reuse the loaded model's own embedded MTP head (Qwen3.5-MTP / Gemma 4); a file = a
    // separate sidecar draft model.
    private void showMtpSelectionDialog() {
        File[] files = getDownloadedProjectorFiles();  // all downloaded GGUFs
        String cur = selectedMtpReference == null ? "" : selectedMtpReference;
        java.util.List<String> labels = new java.util.ArrayList<>();
        final java.util.List<String> refs = new java.util.ArrayList<>();
        boolean ownSel = cur.isEmpty();
        labels.add(localizedText("このモデル自身のMTPヘッドを使用 (推奨)",
                                 "Use this model's own MTP head (recommended)") + (ownSel ? "  ✓" : ""));
        refs.add("");
        for (File f : files) {
            boolean sel = f.getName().equals(cur);
            labels.add(localizedText("別ドラフト: ", "Separate draft: ") + f.getName() + (sel ? "  ✓" : ""));
            refs.add(f.getName());
        }
        new AlertDialog.Builder(this)
                .setTitle(localizedText("MTP ドラフトモデル", "MTP draft model"))
                .setItems(labels.toArray(new String[0]), (dialog, which) -> {
                    selectedMtpReference = refs.get(which);
                    Toast.makeText(this,
                            selectedMtpReference.isEmpty()
                                ? localizedText("MTP: 自モデルのヘッドを使用 (設定保存で反映)",
                                                "MTP: using own head (save config to apply)")
                                : "MTP: " + selectedMtpReference
                                                + localizedText(" (設定保存で反映)", " (save config to apply)"),
                            Toast.LENGTH_LONG).show();
                })
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    private void selectProjectorWithCompatibilityCheck(String modelReference, File projectorFile) {
        if (ModelFileHelper.canAutoApplyProjectorReference(modelReference, projectorFile.getName())) {
            setSelectedProjectorReference(projectorFile.getName(), true);
            return;
        }
        // The chosen mmproj does not look like it matches this model. Warn before applying;
        // an actually-incompatible projector is also disabled automatically at load time (#6).
        new AlertDialog.Builder(this)
                .setTitle(localizedText("mmproj の互換性に注意", "mmproj may be incompatible"))
                .setMessage(localizedText("選択した mmproj (", "The selected mmproj (")
                        + projectorFile.getName()
                        + localizedText(
                        ") はこのモデルと互換でない可能性があります。"
                                + "互換性がない場合、読み込み時に自動で無効化されます。それでも設定しますか？",
                        ") may be incompatible with this model. "
                                + "If it is, it will be disabled automatically at load time. Set it anyway?"))
                .setPositiveButton(localizedText("設定する", "Set anyway"),
                        (dialog, which) -> setSelectedProjectorReference(projectorFile.getName(), true))
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    private void showScrollableSingleChoiceDialog(
            String title,
            String[] items,
            int initialSelection,
            String positiveLabel,
            SelectionHandler onPositive,
            String neutralLabel,
            SelectionHandler onNeutral) {
        showScrollableSingleChoiceDialog(
                title,
                items,
                initialSelection,
                positiveLabel,
                onPositive,
                neutralLabel,
                onNeutral,
                null);
    }

    private void showScrollableSingleChoiceDialog(
            String title,
            String[] items,
            int initialSelection,
            String positiveLabel,
            SelectionHandler onPositive,
            String neutralLabel,
            SelectionHandler onNeutral,
            Runnable onCancel) {
        if (items == null || items.length == 0) {
            return;
        }

        final int[] selectedIndex = {Math.max(0, Math.min(initialSelection, items.length - 1))};
        ListView listView = new ListView(this);
        listView.setChoiceMode(ListView.CHOICE_MODE_SINGLE);
        listView.setAdapter(new ArrayAdapter<>(this, android.R.layout.simple_list_item_single_choice, items));
        listView.setVerticalScrollBarEnabled(true);
        listView.setFastScrollEnabled(true);
        listView.setItemChecked(selectedIndex[0], true);
        listView.setOnItemClickListener((parent, view, position, id) -> selectedIndex[0] = position);

        int rowHeight = (int) (56 * getResources().getDisplayMetrics().density);
        int minHeight = (int) (180 * getResources().getDisplayMetrics().density);
        int maxHeight = (getResources().getDisplayMetrics().heightPixels * 3) / 5;
        int listHeight = Math.min(maxHeight, Math.max(minHeight, items.length * rowHeight));

        LinearLayout container = new LinearLayout(this);
        container.setOrientation(LinearLayout.VERTICAL);
        int padding = (int) (12 * getResources().getDisplayMetrics().density);
        container.setPadding(padding, 0, padding, 0);
        container.addView(listView, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                listHeight));

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle(title)
                .setView(container)
                .setPositiveButton(positiveLabel, (dialogInterface, which) -> onPositive.onSelected(selectedIndex[0]))
                .setNegativeButton(localizedText("閉じる", "Close"), (dialogInterface, which) -> {
                    if (onCancel != null) {
                        onCancel.run();
                    }
                })
                .create();
        if (neutralLabel != null && onNeutral != null) {
            dialog.setButton(AlertDialog.BUTTON_NEUTRAL, neutralLabel,
                    (dialogInterface, which) -> onNeutral.onSelected(selectedIndex[0]));
        }
        dialog.setOnCancelListener(dialogInterface -> {
            if (onCancel != null) {
                onCancel.run();
            }
        });
        dialog.show();
    }

    private void clearIncompatibleProjectorSelection() {
        String modelReference = modelUrlInput != null ? modelUrlInput.getText().toString().trim() : "";
        if (modelReference.isEmpty()) {
            return;
        }
        // When the auto-populated (non-manual) projector no longer matches the current model URL,
        // clear the reference so updateMultimodalProjectorInfo can re-discover for the new model.
        // Also reset selectedProjectorDisabled so the new model gets a fresh auto-discovery pass.
        if (!selectedProjectorReference.isEmpty()
                && !selectedProjectorManualSelection
                && !ModelFileHelper.canAutoApplyProjectorReference(modelReference, selectedProjectorReference)) {
            selectedProjectorReference = "";
            selectedProjectorManualSelection = false;
            selectedProjectorDisabled = false;
        }
    }

    private void setSelectedProjectorReference(String projectorReference) {
        setSelectedProjectorReference(projectorReference, false);
    }

    private void setSelectedProjectorReference(String projectorReference, boolean manualSelection) {
        selectedProjectorReference = normalizeReference(projectorReference);
        selectedProjectorManualSelection = !selectedProjectorReference.isEmpty() && manualSelection;
        // Choosing a projector re-enables vision; an empty reference keeps whatever the caller set
        // (the Clear button sets selectedProjectorDisabled = true explicitly before calling this).
        if (!selectedProjectorReference.isEmpty()) {
            selectedProjectorDisabled = false;
        }
        updateMultimodalProjectorInfo();
    }

    private void updateMultimodalProjectorInfo() {
        if (multimodalProjectorInfo == null) {
            return;
        }
        String modelReference = modelUrlInput != null ? modelUrlInput.getText().toString().trim() : "";
        boolean likelyMultimodalModel =
                !modelReference.isEmpty() && ModelFileHelper.isLikelyMultimodalModelReference(modelReference);

        // When the user explicitly cleared the projector: show the disabled state and stop.
        // Do NOT show the auto-detected file — that would make it look like the clear had no effect.
        if (selectedProjectorDisabled && selectedProjectorReference.isEmpty()) {
            multimodalProjectorInfo.setText(localizedText(
                    "Projector: 解除済み（手動）",
                    "Projector: cleared (manual)"));
            updateActionButtonStateForBusy();
            return;
        }

        // Auto-populate selectedProjectorReference from the co-located mmproj when:
        //   • no explicit selection yet (empty reference)
        //   • user has not disabled it
        //   • a co-located projector file exists on disk
        // Storing the name here makes it explicit so the Clear button becomes enabled,
        // and saving the profile persists the auto-detected choice explicitly —
        // future loads no longer need runtime auto-discovery for this profile.
        if (selectedProjectorReference.isEmpty() && !selectedProjectorDisabled && likelyMultimodalModel) {
            File autoDetected = findAutoDetectedProjectorFile(modelReference);
            if (autoDetected != null) {
                selectedProjectorReference = autoDetected.getName();
                selectedProjectorManualSelection = false;
            }
        }

        if (likelyMultimodalModel) {
            String availableText = localizedText("利用可能", "Available");
            if (!selectedProjectorReference.isEmpty()) {
                multimodalProjectorInfo.setText(
                        availableText + "\n" + extractFilenameFromUrl(selectedProjectorReference));
            } else {
                multimodalProjectorInfo.setText(localizedText(
                        "Projector: 未選択",
                        "Projector: not selected"));
            }
        } else if (selectedProjectorReference.isEmpty()) {
            multimodalProjectorInfo.setText(localizedText(
                    "Projector: 未選択",
                    "Projector: not selected"));
        } else {
            multimodalProjectorInfo.setText(localizedText("Projector: ", "Projector: ")
                    + extractFilenameFromUrl(selectedProjectorReference));
        }
        updateActionButtonStateForBusy();
    }

    // Downloaded model GGUFs (all downloaded GGUFs minus likely mmproj/projector files).
    private File[] getDownloadedModelFiles() {
        File[] all = getDownloadedProjectorFiles();
        List<File> models = new ArrayList<>();
        for (File file : all) {
            if (!ModelFileHelper.isLikelyProjectorFilename(file.getName())) {
                models.add(file);
            }
        }
        return models.toArray(new File[0]);
    }

    // "Select downloaded model": pick from already-downloaded models into the model field.
    private void showDownloadedModelPicker() {
        File[] files = getDownloadedModelFiles();
        if (files.length == 0) {
            showToast(localizedText("ダウンロード済みモデルがありません", "No downloaded models"));
            return;
        }
        String[] names = new String[files.length];
        for (int i = 0; i < files.length; i++) {
            names[i] = files[i].getName();
        }
        new AlertDialog.Builder(this)
                .setTitle(localizedText("モデルを選択", "Select a model"))
                .setItems(names, (dialog, which) -> {
                    modelUrlInput.setText(names[which]);
                    currentConfig = getConfigFromUI();
                    updateAutoTemplatePreview(currentConfig);
                })
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    /** ArrayAdapter that never filters, so the profile field shows the full list on tap. */
    private static class NoFilterArrayAdapter extends ArrayAdapter<String> {
        private final List<String> items;

        NoFilterArrayAdapter(Context context, int resource, List<String> objects) {
            super(context, resource, objects);
            this.items = objects;
        }

        @Override
        public android.widget.Filter getFilter() {
            return new android.widget.Filter() {
                @Override
                protected FilterResults performFiltering(CharSequence constraint) {
                    FilterResults results = new FilterResults();
                    results.values = items;
                    results.count = items.size();
                    return results;
                }

                @Override
                protected void publishResults(CharSequence constraint, FilterResults results) {
                    notifyDataSetChanged();
                }
            };
        }
    }

    private File[] getDownloadedProjectorFiles() {
        List<File> projectorFiles = new ArrayList<>();
        File modelDir = getModelStorageDir();
        File[] existingFiles = modelDir.listFiles();
        if (existingFiles != null) {
            for (File file : existingFiles) {
                if (file.isFile()
                        && file.length() > 0
                        && ModelFileHelper.isGgufFilename(file.getName())
                        && !containsFile(projectorFiles, file)) {
                    projectorFiles.add(file);
                }
            }
        }
        projectorFiles.sort((a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        String modelReference = modelUrlInput != null ? modelUrlInput.getText().toString().trim() : "";
        projectorFiles.sort((a, b) -> {
            boolean aRecommended = ModelFileHelper.canAutoApplyProjectorReference(modelReference, a.getName());
            boolean bRecommended = ModelFileHelper.canAutoApplyProjectorReference(modelReference, b.getName());
            if (aRecommended != bRecommended) {
                return aRecommended ? -1 : 1;
            }
            return a.getName().compareToIgnoreCase(b.getName());
        });
        return projectorFiles.toArray(new File[0]);
    }

    private String normalizeReference(String reference) {
        return reference == null ? "" : reference.trim();
    }

    private ImportedModelCandidate resolveImportedModelCandidate(Uri sourceUri) {
        String displayName = null;
        long sizeBytes = -1L;

        if ("content".equalsIgnoreCase(sourceUri.getScheme())) {
            try (Cursor cursor = getContentResolver().query(
                    sourceUri,
                    new String[]{OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE},
                    null,
                    null,
                    null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                    if (nameIndex >= 0 && !cursor.isNull(nameIndex)) {
                        displayName = cursor.getString(nameIndex);
                    }
                    int sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE);
                    if (sizeIndex >= 0 && !cursor.isNull(sizeIndex)) {
                        sizeBytes = cursor.getLong(sizeIndex);
                    }
                }
            } catch (SecurityException | IllegalArgumentException e) {
                Log.w(TAG, "Failed to query import source metadata", e);
            }
        }

        if (displayName == null || displayName.trim().isEmpty()) {
            displayName = sourceUri.getLastPathSegment();
        }

        displayName = ModelFileHelper.extractFilename(displayName);
        return new ImportedModelCandidate(displayName, sizeBytes);
    }

    private String copyImportedModelToStorage(Uri sourceUri, File destFile, long sizeBytes, String displayName) {
        File parentDir = destFile.getParentFile();
        if (parentDir != null && !parentDir.exists() && !parentDir.mkdirs() && !parentDir.isDirectory()) {
            return "Failed to create model storage directory: " + parentDir.getAbsolutePath();
        }

        File tempFile = new File(
                parentDir != null ? parentDir : getModelStorageDir(),
                destFile.getName() + IMPORT_TEMP_SUFFIX);

        if (tempFile.exists() && !tempFile.delete()) {
            return "Failed to clear temporary import file: " + tempFile.getName();
        }

        try (InputStream inputStream = getContentResolver().openInputStream(sourceUri)) {
            if (inputStream == null) {
                return "Could not open selected file: " + displayName;
            }

            try (OutputStream outputStream = new FileOutputStream(tempFile, false)) {
                byte[] buffer = new byte[MODEL_COPY_BUFFER_SIZE];
                long copiedBytes = 0L;
                int lastProgress = -1;
                int read;
                while ((read = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, read);
                    copiedBytes += read;

                    if (sizeBytes > 0) {
                        int progress = (int) Math.min(100L, (copiedBytes * 100L) / sizeBytes);
                        if (progress != lastProgress) {
                            lastProgress = progress;
                            int progressValue = progress;
                            runOnUiThread(() -> {
                                modelProgressBar.setIndeterminate(false);
                                modelProgressBar.setProgress(progressValue);
                                modelFileInfo.setText(localizedText(
                                        "モデルを取り込み中... ",
                                        "Importing model... ") + displayName + " (" + progressValue + "%)");
                            });
                        }
                    }
                }
                outputStream.flush();
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to import model file", e);
            if (tempFile.exists() && !tempFile.delete()) {
                Log.w(TAG, "Failed to delete temporary import file: " + tempFile.getAbsolutePath());
            }
            return "Failed to import model file: " + e.getMessage();
        }

        if (!tempFile.exists() || tempFile.length() <= 0) {
            if (tempFile.exists() && !tempFile.delete()) {
                Log.w(TAG, "Failed to delete empty import file: " + tempFile.getAbsolutePath());
            }
            return "Imported model file is empty: " + displayName;
        }

        if (destFile.exists() && !destFile.delete()) {
            if (!tempFile.delete()) {
                Log.w(TAG, "Failed to delete temporary import file after replace failure: " + tempFile.getAbsolutePath());
            }
            return "Failed to replace existing model file: " + destFile.getName();
        }

        if (!tempFile.renameTo(destFile)) {
            if (!tempFile.delete()) {
                Log.w(TAG, "Failed to delete temporary import file after rename failure: " + tempFile.getAbsolutePath());
            }
            return "Failed to finalize imported model file: " + destFile.getName();
        }

        return null;
    }
    
    private void loadModel(boolean allowProjectorDownload) {
        final ConfigurationManager.Configuration config = saveConfigurationForModelAction();
        if (config == null) {
            return;
        }

        if (modelManager.isBusy()) {
            showToast(localizedText("他のリクエストを処理中です", "Model is busy processing another request"));
            return;
        }

        new Thread(() -> {
            // Acquire busy lock for load
            if (!modelManager.tryAcquire()) {
                runOnUiThread(() -> showToast(localizedText("モデルは処理中です", "Model is busy")));
                return;
            }

            try {
                boolean success = modelManager.loadConfiguration(config.name, false, false, allowProjectorDownload);
                final String disabledMmproj = modelManager.consumeLastDisabledMmprojMessage();
                runOnUiThread(() -> {
                    loadedModelPath = modelManager.getCurrentModelPath();
                    modelLoadedSuccessfully = success;
                    modelFileInfo.setText(success
                        ? "Model loaded: " + (loadedModelPath == null ? config.name : new File(loadedModelPath).getName())
                        : "Model load failed");
                    modelProgressBar.setProgress(success ? 100 : 0);
                    lastDownloadProgress = success ? 100 : 0;
                    loadModelButton.setEnabled(true);
                    showToast(success
                            ? localizedText("モデルの初期化に成功しました", "Model initialized successfully")
                            : localizedText("モデルの初期化に失敗しました", "Model initialization failed"));
                    if (success && disabledMmproj != null) {
                        showToast(localizedText(
                                "選択した mmproj はこのモデルと非互換のため無効化し、テキスト専用で読み込みました: ",
                                "The selected mmproj is incompatible and was disabled; loaded text-only: ") + disabledMmproj);
                    }
                    updateAutoTemplatePreview(config);
                });
            } catch (Throwable t) {
                Log.e(TAG, "Model load error", t);
                runOnUiThread(() -> {
                    showToast(localizedText("モデル読み込みエラー: ", "Model load error: ") + t.getMessage());
                    modelFileInfo.setText(localizedText("モデルの初期化に失敗しました", "Model init failed"));
                    modelProgressBar.setProgress(0);
                    lastDownloadProgress = 0;
                    loadModelButton.setEnabled(true);
                });
            } finally {
                modelManager.release();
            }
        }).start();
    }

    private void startModelAction(boolean loadAfterDownload, Runnable onCancel) {
        ConfigurationManager.Configuration config = getConfigFromUI();
        if (requiresRemoteDownload(config)) {
            showRemoteDownloadDecisionDialog(loadAfterDownload, onCancel);
            return;
        }

        // Nothing remote to fetch -> no separate mmproj download to confirm.
        runModelAction(loadAfterDownload, true);
    }

    /**
     * After the model download/load decision is made, ask the user before downloading a
     * separate remote mmproj (request #2). Skipped when the model itself is an mmproj
     * (request #5) or when no remote mmproj download is pending.
     */
    private void startResolvedModelAction(boolean loadAfterDownload) {
        ConfigurationManager.Configuration config = getConfigFromUI();
        if (!mmprojDownloadNeedsConfirmation(config)) {
            runModelAction(loadAfterDownload, true);
            return;
        }

        String mmprojName = extractFilenameFromUrl(config.multimodalProjectorUrl);
        new AlertDialog.Builder(this)
                .setTitle(localizedText("mmproj をダウンロード", "Download mmproj"))
                .setMessage(localizedText("このマルチモーダルモデルは mmproj (", "This multimodal model also uses an mmproj (")
                        + mmprojName
                        + localizedText(
                        ") も使用します。"
                                + "続けてダウンロードしますか？\nスキップした場合はテキスト専用で読み込みます。",
                        "). "
                                + "Download it as well?\nIf you skip, the model loads text-only."))
                .setPositiveButton(localizedText("ダウンロードする", "Download"),
                        (dialog, which) -> runModelAction(loadAfterDownload, true))
                .setNegativeButton(localizedText("スキップ", "Skip"),
                        (dialog, which) -> runModelAction(loadAfterDownload, false))
                .show();
    }

    private void runModelAction(boolean loadAfterDownload, boolean allowProjectorDownload) {
        if (loadAfterDownload) {
            loadModel(allowProjectorDownload);
        } else {
            downloadModelFilesOnly(allowProjectorDownload);
        }
    }

    private boolean mmprojDownloadNeedsConfirmation(ConfigurationManager.Configuration config) {
        // The model itself being an mmproj means there is no separate projector to fetch.
        String modelFilename = extractFilenameFromUrl(config.modelUrl);
        if (ModelFileHelper.isLikelyProjectorFilename(modelFilename)) {
            return false;
        }
        return referenceNeedsRemoteDownload(config.multimodalProjectorUrl);
    }

    private void showRemoteDownloadDecisionDialog(boolean loadAfterDownload, Runnable onCancel) {
        ConfigurationManager.Configuration config = getConfigFromUI();
        String modelFilename = extractFilenameFromUrl(config.modelUrl);
        boolean projectorDownload = ModelFileHelper.isLikelyProjectorFilename(modelFilename);
        boolean hasRemoteProjector = referenceNeedsRemoteDownload(config.multimodalProjectorUrl);

        String title = localizedText(
                projectorDownload ? "Projector をダウンロード" : "モデルをダウンロード",
                projectorDownload ? "Download projector" : "Download model");
        String message;
        if (projectorDownload) {
            message = localizedText(
                    "この GGUF は mmproj / Projector の可能性があります。ダウンロードのみを推奨します。ダウンロード後にそのままロードしますか？",
                    "This GGUF looks like an mmproj / projector. Download-only is recommended. Load it immediately after the download?");
        } else if (hasRemoteProjector) {
            message = localizedText(
                    "この操作ではモデル本体と利用可能な Projector をダウンロードします。ダウンロード後にモデルをロードしますか？",
                    "This will download the model and an available projector. Load the model immediately after the download?");
        } else {
            message = localizedText(
                    "この操作では Web からモデルをダウンロードします。ダウンロード後にすぐロードしますか？",
                    "This will download the model from the web. Load it immediately after the download?");
        }

        AlertDialog.Builder builder = new AlertDialog.Builder(this)
                .setTitle(title)
                .setMessage(message)
                .setNegativeButton(localizedText("キャンセル", "Cancel"), (dialog, which) -> {
                    if (onCancel != null) {
                        onCancel.run();
                    }
                });

        if (projectorDownload) {
            builder.setPositiveButton(
                    localizedText("ダウンロードのみ（推奨）", "Download only (Recommended)"),
                    (dialog, which) -> startResolvedModelAction(false));
            builder.setNeutralButton(
                    localizedText("ダウンロードしてロード", "Download and load"),
                    (dialog, which) -> startResolvedModelAction(true));
        } else if (loadAfterDownload) {
            builder.setPositiveButton(
                    localizedText("ダウンロードしてロード", "Download and load"),
                    (dialog, which) -> startResolvedModelAction(true));
            builder.setNeutralButton(
                    localizedText("ダウンロードのみ", "Download only"),
                    (dialog, which) -> startResolvedModelAction(false));
        } else {
            builder.setPositiveButton(
                    localizedText("ダウンロードのみ", "Download only"),
                    (dialog, which) -> startResolvedModelAction(false));
            builder.setNeutralButton(
                    localizedText("ダウンロードしてロード", "Download and load"),
                    (dialog, which) -> startResolvedModelAction(true));
        }

        AlertDialog dialog = builder.create();
        dialog.setOnCancelListener(ignored -> {
            if (onCancel != null) {
                onCancel.run();
            }
        });
        dialog.show();
    }

    private ConfigurationManager.Configuration saveConfigurationForModelAction() {
        final ConfigurationManager.Configuration config = getConfigFromUI();
        try {
            configManager.saveConfiguration(config);
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to save configuration before model action", e);
            showToast(localizedText("設定の保存に失敗しました: ", "Failed to save configuration: ") + e.getMessage());
            return null;
        }

        final String filename = extractFilenameFromUrl(config.modelUrl);
        if (filename != null && !filename.isEmpty()) {
            File destFile = new File(getModelStorageDir(), filename);
            modelFileInfo.setText(localizedText("モデルファイル: ", "Model file: ") + filename + (destFile.exists()
                    ? " (" + destFile.length() + " bytes)"
                    : localizedText(" (確認中...)", " (checking...)")));
        } else {
            modelFileInfo.setText(localizedText("モデルファイル: （不明）", "Model file: (unknown)"));
        }
        modelProgressBar.setProgress(0);
        lastDownloadProgress = 0;
        return config;
    }

    private void downloadModelFilesOnly(boolean allowProjectorDownload) {
        final ConfigurationManager.Configuration config = saveConfigurationForModelAction();
        if (config == null) {
            return;
        }

        if (modelManager.isBusy()) {
            showToast(localizedText("他のリクエストを処理中です", "Model is busy processing another request"));
            return;
        }

        new Thread(() -> {
            if (!modelManager.tryAcquire()) {
                runOnUiThread(() -> showToast(localizedText("モデルは処理中です", "Model is busy")));
                return;
            }

            try {
                boolean success = modelManager.downloadConfigurationAssets(config.name, allowProjectorDownload);
                runOnUiThread(() -> {
                    String filename = extractFilenameFromUrl(config.modelUrl);
                    modelFileInfo.setText(success
                            ? localizedText("ダウンロード完了: ", "Download complete: ")
                                    + (filename == null ? config.name : filename)
                            : localizedText("ダウンロードに失敗しました", "Download failed"));
                    modelProgressBar.setProgress(success ? 100 : 0);
                    lastDownloadProgress = success ? 100 : 0;
                    updateAutoTemplatePreview(config);
                    showToast(success
                            ? localizedText("ダウンロードが完了しました", "Download completed")
                            : localizedText("ダウンロードに失敗しました", "Download failed"));
                });
            } catch (Throwable t) {
                Log.e(TAG, "Model download error", t);
                runOnUiThread(() -> {
                    showToast(localizedText("モデルのダウンロードエラー: ", "Model download error: ") + t.getMessage());
                    modelFileInfo.setText(localizedText("ダウンロードに失敗しました", "Download failed"));
                    modelProgressBar.setProgress(0);
                    lastDownloadProgress = 0;
                });
            } finally {
                modelManager.release();
            }
        }).start();
    }

    private boolean requiresRemoteDownload(ConfigurationManager.Configuration config) {
        return referenceNeedsRemoteDownload(config.modelUrl)
                || referenceNeedsRemoteDownload(config.multimodalProjectorUrl);
    }

    private boolean referenceNeedsRemoteDownload(String reference) {
        String normalizedReference = normalizeReference(reference);
        if (normalizedReference.isEmpty() || !ModelFileHelper.isRemoteModelReference(normalizedReference)) {
            return false;
        }
        File storedFile = ModelFileHelper.resolveStoredModelFile(this, normalizedReference);
        return storedFile == null || !storedFile.isFile() || storedFile.length() <= 0;
    }

    private File findAutoDetectedProjectorFile(String modelReference) {
        String normalizedModelReference = normalizeReference(modelReference);
        if (normalizedModelReference.isEmpty()) {
            return null;
        }
        return ModelFileHelper.findAutoDetectedMultimodalProjectorFile(
                this,
                normalizedModelReference,
                false,
                false);
    }
    
    private void initModelInBackground(final String modelPath) {
        runOnUiThread(() -> {
            modelFileInfo.setText(localizedText("モデルを初期化中...", "Initializing model..."));
            modelProgressBar.setProgress(0);
            loadModelButton.setEnabled(false);
        });
        
        new Thread(() -> {
            // Try to acquire the lock
            if (!modelManager.tryAcquire()) {
                runOnUiThread(() -> {
                    showToast(localizedText("モデルは処理中です", "Model is busy"));
                    loadModelButton.setEnabled(true);
                });
                return;
            }
            
            try {
                LlamaNative llama = modelManager.getLlama();
                String initResult = llama.init(modelPath);
                
                if (!"ok".equals(initResult)) {
                    runOnUiThread(() -> {
                        showToast(localizedText("モデルの初期化に失敗しました: ", "Model init failed: ") + initResult);
                        modelFileInfo.setText(localizedText("モデルの初期化に失敗しました: ", "Model init failed: ") + initResult);
                        loadModelButton.setEnabled(true);
                    });
                    return;
                }
                
                // Set parameters after successful model initialization
                ConfigurationManager.Configuration config = getConfigFromUI();
                modelManager.applyConfiguration(config);
                
                runOnUiThread(() -> {
                    loadedModelPath = modelPath;
                    modelLoadedSuccessfully = true;
                    modelFileInfo.setText(localizedText("モデルを読み込みました: ", "Model loaded: ") + (new File(modelPath).getName()));
                    loadModelButton.setEnabled(true);
                    modelProgressBar.setProgress(100);
                    showToast(localizedText("モデルの初期化に成功しました", "Model initialized successfully"));
                    updateAutoTemplatePreview(config);
                });
            } catch (Throwable t) {
                Log.e(TAG, "Model init error", t);
                runOnUiThread(() -> {
                    showToast(localizedText("モデル初期化エラー: ", "Model init error: ") + t.getMessage());
                    modelFileInfo.setText(localizedText("モデルの初期化に失敗しました", "Model init failed"));
                    loadModelButton.setEnabled(true);
                });
            } finally {
                modelManager.release();
            }
        }).start();
    }
    
    private String extractFilenameFromUrl(String url) {
        return ModelFileHelper.extractFilename(url);
    }

    private File getModelStorageDir() {
        return ModelFileHelper.getModelStorageDir(this);
    }
    
    private void performBackup() {
        String ts = String.format(Locale.US, "%1$tY-%1$tm-%1$td_%1$tH%1$tM%1$tS", new java.util.Date());
        String folderName = "LlamaBackup_" + ts;
        showToast(localizedText("バックアップを開始しています...", "Starting backup..."));
        new Thread(() -> {
            try {
                File configDir = new File(getExternalFilesDir(null), "configs");
                File[] configFiles = configDir.listFiles((d, n) -> n.endsWith(".json"));
                File[] modelFiles = getManagedGgufFiles();
                int profileCount = 0, modelCount = 0;
                String settingsJson = buildSharedSettingsJson().toString(2);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    String root = "Download/" + folderName + "/";
                    insertJsonToDownloads(settingsJson, "settings.json", root);
                    if (configFiles != null) {
                        for (File f : configFiles) {
                            if (insertToDownloads(f, root + "profiles/", "application/json")) profileCount++;
                        }
                    }
                    for (File f : modelFiles) {
                        if (insertToDownloads(f, root + "models/", "application/octet-stream")) modelCount++;
                    }
                    final int pc = profileCount, mc = modelCount;
                    showToast(localizedText(
                        "バックアップ完了: Download/" + folderName + " (" + pc + " プロファイル, " + mc + " モデル)",
                        "Backup complete: Download/" + folderName + " (" + pc + " profiles, " + mc + " models)"
                    ));
                } else {
                    File backupDir = new File(getExternalFilesDir(null), "backups/" + folderName);
                    File profilesDir = new File(backupDir, "profiles");
                    File modelsDir = new File(backupDir, "models");
                    profilesDir.mkdirs();
                    modelsDir.mkdirs();
                    writeTextFile(new File(backupDir, "settings.json"), settingsJson);
                    if (configFiles != null) {
                        for (File f : configFiles) { streamCopyFile(f, new File(profilesDir, f.getName())); profileCount++; }
                    }
                    for (File f : modelFiles) { streamCopyFile(f, new File(modelsDir, f.getName())); modelCount++; }
                    final int pc = profileCount, mc = modelCount;
                    showToast(localizedText(
                        "バックアップ完了: " + backupDir.getAbsolutePath() + " (" + pc + " プロファイル, " + mc + " モデル)",
                        "Backup complete: " + backupDir.getAbsolutePath() + " (" + pc + " profiles, " + mc + " models)"
                    ));
                }
            } catch (Exception e) {
                Log.e(TAG, "Backup failed", e);
                showToast(localizedText("バックアップに失敗しました: ", "Backup failed: ") + e.getMessage());
            }
        }).start();
    }

    private org.json.JSONObject buildSharedSettingsJson() throws JSONException {
        org.json.JSONObject json = new org.json.JSONObject();
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        String lang = prefs.getString("display_language", null);
        if (lang != null) json.put("display_language", lang);
        json.put(PREF_API_PORT, prefs.getInt(PREF_API_PORT, OllamaApiServer.DEFAULT_PORT));
        json.put(PREF_BUSY_QUEUE_WAIT_SECONDS, prefs.getInt(PREF_BUSY_QUEUE_WAIT_SECONDS, DEFAULT_BUSY_QUEUE_WAIT_SECONDS));
        json.put(OllamaForegroundService.PREF_KEEP_AWAKE, prefs.getBoolean(OllamaForegroundService.PREF_KEEP_AWAKE, false));
        if (prefs.contains(PREF_LOG_LEVEL)) json.put(PREF_LOG_LEVEL, prefs.getInt(PREF_LOG_LEVEL, 2));
        json.put(PREF_SHOW_PERF_METRICS, prefs.getBoolean(PREF_SHOW_PERF_METRICS, false));
        json.put(McpSettingsHelper.PREF_SHARED_MCP_SERVERS_JSON, McpSettingsHelper.getSharedMcpServersJson(this));
        json.put(McpSettingsHelper.PREF_SHARED_FUNCTION_DEFINITIONS_JSON, McpSettingsHelper.getSharedFunctionDefinitionsJson(this));
        json.put(McpSettingsHelper.PREF_SHARED_MCP_ENABLED, McpSettingsHelper.isSharedMcpEnabledOutsideWebUi(this));
        json.put(McpSettingsHelper.PREF_SHARED_FUNCTION_DEFINITIONS_ENABLED, McpSettingsHelper.isSharedFunctionDefinitionsEnabledOutsideWebUi(this));
        return json;
    }

    private void applySharedSettingsJson(org.json.JSONObject json) {
        SharedPreferences.Editor ed = getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit();
        if (json.has("display_language")) ed.putString("display_language", json.optString("display_language"));
        if (json.has(PREF_API_PORT)) ed.putInt(PREF_API_PORT, json.optInt(PREF_API_PORT));
        if (json.has(PREF_BUSY_QUEUE_WAIT_SECONDS)) ed.putInt(PREF_BUSY_QUEUE_WAIT_SECONDS, json.optInt(PREF_BUSY_QUEUE_WAIT_SECONDS));
        if (json.has(OllamaForegroundService.PREF_KEEP_AWAKE)) ed.putBoolean(OllamaForegroundService.PREF_KEEP_AWAKE, json.optBoolean(OllamaForegroundService.PREF_KEEP_AWAKE));
        if (json.has(PREF_LOG_LEVEL)) ed.putInt(PREF_LOG_LEVEL, json.optInt(PREF_LOG_LEVEL));
        if (json.has(PREF_SHOW_PERF_METRICS)) ed.putBoolean(PREF_SHOW_PERF_METRICS, json.optBoolean(PREF_SHOW_PERF_METRICS));
        ed.apply();
        if (json.has(McpSettingsHelper.PREF_SHARED_MCP_SERVERS_JSON))
            McpSettingsHelper.saveSharedMcpServersJson(this, json.optString(McpSettingsHelper.PREF_SHARED_MCP_SERVERS_JSON));
        if (json.has(McpSettingsHelper.PREF_SHARED_FUNCTION_DEFINITIONS_JSON))
            McpSettingsHelper.saveSharedFunctionDefinitionsJson(this, json.optString(McpSettingsHelper.PREF_SHARED_FUNCTION_DEFINITIONS_JSON));
        if (json.has(McpSettingsHelper.PREF_SHARED_MCP_ENABLED))
            McpSettingsHelper.saveSharedMcpEnabledOutsideWebUi(this, json.optBoolean(McpSettingsHelper.PREF_SHARED_MCP_ENABLED));
        if (json.has(McpSettingsHelper.PREF_SHARED_FUNCTION_DEFINITIONS_ENABLED))
            McpSettingsHelper.saveSharedFunctionDefinitionsEnabledOutsideWebUi(this, json.optBoolean(McpSettingsHelper.PREF_SHARED_FUNCTION_DEFINITIONS_ENABLED));
    }

    @android.annotation.SuppressLint("InlinedApi")
    private boolean insertToDownloads(File src, String relativePath, String mimeType) throws IOException {
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.Downloads.DISPLAY_NAME, src.getName());
        cv.put(MediaStore.Downloads.RELATIVE_PATH, relativePath);
        cv.put(MediaStore.Downloads.MIME_TYPE, mimeType);
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv);
        if (uri == null) return false;
        try (OutputStream os = getContentResolver().openOutputStream(uri);
             InputStream is = new FileInputStream(src)) {
            streamCopy(is, os);
        }
        return true;
    }

    @android.annotation.SuppressLint("InlinedApi")
    private void insertJsonToDownloads(String content, String filename, String relativePath) throws IOException {
        ContentValues cv = new ContentValues();
        cv.put(MediaStore.Downloads.DISPLAY_NAME, filename);
        cv.put(MediaStore.Downloads.RELATIVE_PATH, relativePath);
        cv.put(MediaStore.Downloads.MIME_TYPE, "application/json");
        Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv);
        if (uri == null) return;
        try (OutputStream os = getContentResolver().openOutputStream(uri)) {
            os.write(content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }
    }

    private void writeTextFile(File dst, String content) throws IOException {
        try (OutputStream os = new FileOutputStream(dst)) {
            os.write(content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }
    }

    private void performRestore() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        startActivityForResult(intent, REQUEST_RESTORE_DIR);
    }

    private void restoreFromTree(Uri treeUri) {
        showToast(localizedText("復元を開始しています...", "Starting restore..."));
        new Thread(() -> {
            try {
                int profileCount = 0, modelCount = 0;
                String rootDocId = DocumentsContract.getTreeDocumentId(treeUri);
                Uri rootChildrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, rootDocId);
                String[] proj = {
                    DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                    DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                    DocumentsContract.Document.COLUMN_MIME_TYPE
                };

                String profilesFolderId = null, modelsFolderId = null, settingsDocId = null;
                try (Cursor c = getContentResolver().query(rootChildrenUri, proj, null, null, null)) {
                    while (c != null && c.moveToNext()) {
                        String id = c.getString(0), name = c.getString(1), mime = c.getString(2);
                        if ("profiles".equals(name) && DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) profilesFolderId = id;
                        if ("models".equals(name) && DocumentsContract.Document.MIME_TYPE_DIR.equals(mime)) modelsFolderId = id;
                        if ("settings.json".equals(name)) settingsDocId = id;
                    }
                }

                // Restore shared settings
                if (settingsDocId != null) {
                    Uri settingsUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, settingsDocId);
                    try (InputStream is = getContentResolver().openInputStream(settingsUri);
                         java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream()) {
                        if (is != null) {
                            byte[] buf = new byte[4096];
                            int n;
                            while ((n = is.read(buf)) != -1) baos.write(buf, 0, n);
                            org.json.JSONObject json = new org.json.JSONObject(baos.toString("UTF-8"));
                            runOnUiThread(() -> applySharedSettingsJson(json));
                        }
                    } catch (Exception e) {
                        Log.w(TAG, "Failed to restore settings.json", e);
                    }
                }

                File configDir = new File(getExternalFilesDir(null), "configs");
                if (profilesFolderId != null) {
                    Uri childUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, profilesFolderId);
                    try (Cursor c = getContentResolver().query(childUri, proj, null, null, null)) {
                        while (c != null && c.moveToNext()) {
                            String id = c.getString(0), name = c.getString(1);
                            if (name.endsWith(".json")) {
                                Uri fileUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, id);
                                try (InputStream is = getContentResolver().openInputStream(fileUri);
                                     OutputStream os = new FileOutputStream(new File(configDir, name))) {
                                    streamCopy(is, os);
                                }
                                profileCount++;
                            }
                        }
                    }
                }

                File modelDir = ModelFileHelper.getModelStorageDir(this);
                if (modelsFolderId != null) {
                    Uri childUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, modelsFolderId);
                    try (Cursor c = getContentResolver().query(childUri, proj, null, null, null)) {
                        while (c != null && c.moveToNext()) {
                            String id = c.getString(0), name = c.getString(1);
                            if (ModelFileHelper.isGgufFilename(name)) {
                                Uri fileUri = DocumentsContract.buildDocumentUriUsingTree(treeUri, id);
                                File dest = new File(modelDir, name);
                                if (!dest.exists()) {
                                    try (InputStream is = getContentResolver().openInputStream(fileUri);
                                         OutputStream os = new FileOutputStream(dest)) {
                                        streamCopy(is, os);
                                    }
                                    modelCount++;
                                }
                            }
                        }
                    }
                }

                final int pc = profileCount, mc = modelCount;
                final boolean restoredSettings = settingsDocId != null;
                showToast(localizedText(
                    "復元完了: " + pc + " プロファイル, " + mc + " モデル" + (restoredSettings ? ", 共通設定" : ""),
                    "Restore complete: " + pc + " profiles, " + mc + " models" + (restoredSettings ? ", shared settings" : "")
                ));
                runOnUiThread(() -> loadConfigList());
            } catch (Exception e) {
                Log.e(TAG, "Restore failed", e);
                showToast(localizedText("復元に失敗しました: ", "Restore failed: ") + e.getMessage());
            }
        }).start();
    }

    private void streamCopy(InputStream is, OutputStream os) throws IOException {
        byte[] buf = new byte[65536];
        int n;
        while ((n = is.read(buf)) != -1) os.write(buf, 0, n);
    }

    private void streamCopyFile(File src, File dst) throws IOException {
        try (InputStream is = new FileInputStream(src);
             OutputStream os = new FileOutputStream(dst)) {
            streamCopy(is, os);
        }
    }

    private void showToast(final String msg) {
        runOnUiThread(() -> {
            Toast toast = Toast.makeText(SettingsActivity.this, msg, Toast.LENGTH_LONG);
            toast.setGravity(Gravity.CENTER, 0, 0);
            toast.show();
        });
    }
    
    /** Cancel without saving — simply close the activity. */
    private void cancelAndReturn() {
        setResult(RESULT_CANCELED);
        super.finish();
    }

    @Override
    protected void onResume() {
        super.onResume();
        registerNetworkCallback();
        updateActionButtonStateForBusy();
        updateApiServerUrlViews();
    }

    @Override
    protected void onPause() {
        super.onPause();
        unregisterNetworkCallback();
    }

    @Override
    protected void onDestroy() {
        if (modelManager != null) {
            modelManager.removeBusyStateListener(busyStateListener);
        }
        super.onDestroy();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == REQUEST_RESTORE_DIR) {
            if (resultCode == RESULT_OK && data != null) {
                Uri treeUri = data.getData();
                if (treeUri != null) {
                    try {
                        getContentResolver().takePersistableUriPermission(treeUri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
                    } catch (SecurityException e) {
                        Log.w(TAG, "Cannot persist read permission for restore dir", e);
                    }
                    restoreFromTree(treeUri);
                }
            }
            return;
        }

        if (requestCode == REQUEST_IMPORT_LORA_ADAPTER) {
            if (resultCode != RESULT_OK) {
                return;
            }
            Uri loraUri = data != null ? data.getData() : null;
            if (loraUri == null) {
                showToast(localizedText("選択したファイルを開けません", "Could not open the selected file"));
                return;
            }
            importAndApplyLoraAdapter(loraUri);
            return;
        }

        if (requestCode != REQUEST_IMPORT_MODEL_LOCAL_DEVICE || resultCode != RESULT_OK) {
            return;
        }

        Uri selectedUri = data != null ? data.getData() : null;
        if (selectedUri == null) {
            showToast(localizedText("選択したファイルを開けません", "Could not open the selected file"));
            return;
        }

        int grantedFlags = data != null ? data.getFlags() : 0;
        int persistableReadFlags = grantedFlags & Intent.FLAG_GRANT_READ_URI_PERMISSION;
        if (persistableReadFlags != 0) {
            try {
                getContentResolver().takePersistableUriPermission(selectedUri, persistableReadFlags);
            } catch (SecurityException e) {
                Log.w(TAG, "Persistable read permission unavailable for imported model URI", e);
            }
        }

        importModelFromUri(selectedUri);
    }

    // ---- LoRA アダプタ適用 ----------------------------------------------------

    /** 選択されたアダプタGGUFをアプリ領域へコピーし、スケールを尋ねてから適用する。
     *  ネイティブは実ファイルパスで開くため、SAF Uri は一旦コピーが必要。 */
    private void importAndApplyLoraAdapter(Uri sourceUri) {
        if (!modelLoadedSuccessfully) {
            showToast(localizedText("先にモデルを読み込んでください", "Load a model first"));
            return;
        }
        String displayName = resolveImportedModelCandidate(sourceUri).displayName;
        final String name = (displayName != null && ModelFileHelper.isGgufFilename(displayName))
                ? displayName
                : ((displayName != null && !displayName.isEmpty() ? displayName : "adapter") + ".gguf");

        // スケール入力ダイアログ（既定 1.0）
        final EditText scaleInput = new EditText(this);
        scaleInput.setInputType(android.text.InputType.TYPE_CLASS_NUMBER | android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL);
        scaleInput.setText("1.0");
        new AlertDialog.Builder(this)
                .setTitle(localizedText("LoRA スケール", "LoRA scale"))
                .setMessage(localizedText("適用強度（通常 1.0）", "Effective strength (usually 1.0)"))
                .setView(scaleInput)
                .setPositiveButton(localizedText("適用", "Apply"), (d, w) -> {
                    float scale;
                    try { scale = Float.parseFloat(scaleInput.getText().toString().trim()); }
                    catch (Exception e) { scale = 1.0f; }
                    doImportAndApplyLora(sourceUri, name, scale);
                })
                .setNegativeButton(localizedText("キャンセル", "Cancel"), null)
                .show();
    }

    private void doImportAndApplyLora(Uri sourceUri, String name, float scale) {
        runOnUiThread(() -> {
            loraAdapterInfo.setText(localizedText("アダプタ取込中... ", "Importing adapter... ") + name);
            applyLoraButton.setEnabled(false);
        });
        new Thread(() -> {
            String error = null;
            String destPath = null;
            try {
                File dir = new File(getFilesDir(), "adapters");
                if (!dir.exists()) dir.mkdirs();
                File dest = new File(dir, name);
                try (java.io.InputStream in = getContentResolver().openInputStream(sourceUri);
                     java.io.OutputStream out = new java.io.FileOutputStream(dest)) {
                    if (in == null) throw new java.io.IOException("openInputStream=null");
                    byte[] buf = new byte[MODEL_COPY_BUFFER_SIZE];
                    int n;
                    while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
                }
                destPath = dest.getAbsolutePath();
                if (!modelManager.tryAcquire()) {
                    error = localizedText("モデルは処理中です", "Model is busy");
                } else {
                    try {
                        error = modelManager.getLlama().applyLoraAdapter(destPath, scale);
                    } finally {
                        modelManager.release();
                    }
                }
            } catch (Throwable t) {
                error = t.getMessage();
            }
            final String fError = error;
            final float fScale = scale;
            final String fName = name;
            runOnUiThread(() -> {
                applyLoraButton.setEnabled(true);
                if (fError == null || fError.isEmpty()) {
                    loraAdapterInfo.setText(localizedText("適用中: ", "Applied: ") + fName
                            + "  (scale=" + fScale + ")");
                    showToast(localizedText("LoRAアダプタを適用しました", "LoRA adapter applied"));
                } else {
                    loraAdapterInfo.setText(localizedText("適用失敗: ", "Apply failed: ") + fError);
                    showToast(localizedText("LoRA適用失敗: ", "LoRA apply failed: ") + fError);
                }
            });
        }).start();
    }

    private void clearLoraAdapter() {
        new Thread(() -> {
            boolean acquired = modelManager.tryAcquire();
            try {
                if (acquired) modelManager.getLlama().clearLoraAdapter();
            } catch (Throwable ignored) {
            } finally {
                if (acquired) modelManager.release();
            }
            runOnUiThread(() -> {
                loraAdapterInfo.setText(localizedText("LoRA アダプタ未適用", "No LoRA adapter"));
                showToast(localizedText("LoRAアダプタを解除しました", "LoRA adapter cleared"));
            });
        }).start();
    }

    @Override
    public void finish() {
        if (!saveSharedSettings()) {
            return;
        }
        int apiPort = resolveApiPortFromUi();
        
        // Save current UI configuration before returning
        ConfigurationManager.Configuration config = getConfigFromUI();
        try {
            configManager.saveConfiguration(config);
            currentConfig = config;
        } catch (IOException | JSONException e) {
            Log.e(TAG, "Failed to save configuration on finish", e);
        }
        
        // Return the current configuration name and model info to MainActivity
        Intent resultIntent = new Intent();
        if (currentConfig != null) {
            resultIntent.putExtra(EXTRA_CONFIG_NAME, currentConfig.name);
        }
        if (loadedModelPath != null) {
            resultIntent.putExtra(EXTRA_MODEL_PATH, loadedModelPath);
            resultIntent.putExtra(EXTRA_MODEL_LOADED, modelLoadedSuccessfully);
        }
        resultIntent.putExtra(EXTRA_API_PORT, apiPort);
        resultIntent.putExtra(EXTRA_DISPLAY_LANGUAGE, AppLanguageManager.getOrInitDisplayLanguage(this));
        setResult(RESULT_OK, resultIntent);
        super.finish();
    }
}
