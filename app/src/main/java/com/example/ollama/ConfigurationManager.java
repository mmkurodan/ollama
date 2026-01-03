package com.example.ollama;

import android.content.Context;
import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ConfigurationManager {
    private static final String TAG = "ConfigurationManager";
    private static final String CONFIG_DIR = "configs";
    private static final String DEFAULT_CONFIG_NAME = "default";
    
    private final Context context;
    private final File configDir;
    
    public static class Configuration {
        public String name;
        public String modelUrl;
        public int nCtx;
        public int nThreads;
        public int nBatch;
        public double temp;
        public double topP;
        public int topK;
        public String promptTemplate;
        
        // Penalty parameters
        public int penaltyLastN;
        public double penaltyRepeat;
        public double penaltyFreq;
        public double penaltyPresent;
        
        // Mirostat parameters
        public int mirostat;
        public double mirostatTau;
        public double mirostatEta;
        
        // Additional sampling parameters
        public double minP;
        public double typicalP;
        public double dynatempRange;
        public double dynatempExponent;
        public double xtcProbability;
        public double xtcThreshold;
        public double topNSigma;
        
        // DRY parameters
        public double dryMultiplier;
        public double dryBase;
        public int dryAllowedLength;
        public int dryPenaltyLastN;
        public String drySequenceBreakers;
        
        public Configuration() {
            // Default values
            name = DEFAULT_CONFIG_NAME;
            modelUrl = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
            nCtx = 2048;
            nThreads = 2;
            nBatch = 16;
            temp = 0.7;
            topP = 0.9;
            topK = 40;
            promptTemplate = "<|system|>\nYou are a helpful assistant.\n<|user|>\n{USER_INPUT}\n<|assistant|>\n";
            
            // Penalty parameters defaults
            penaltyLastN = 64;
            penaltyRepeat = 1.0;
            penaltyFreq = 0.0;
            penaltyPresent = 0.0;
            
            // Mirostat parameters defaults
            mirostat = 0;
            mirostatTau = 5.0;
            mirostatEta = 0.1;
            
            // Additional sampling parameters defaults
            minP = 0.05;
            typicalP = 1.0;
            dynatempRange = 0.0;
            dynatempExponent = 1.0;
            xtcProbability = 0.0;
            xtcThreshold = 0.1;
            topNSigma = -1.0;
            
            // DRY parameters defaults
            dryMultiplier = 0.0;
            dryBase = 1.75;
            dryAllowedLength = 2;
            dryPenaltyLastN = -1;
            drySequenceBreakers = "\\n,:,\",*";
        }
        
        public Configuration(String name) {
            this();
            this.name = name;
        }
        
        public JSONObject toJSON() throws JSONException {
            JSONObject json = new JSONObject();
            json.put("name", name);
            json.put("modelUrl", modelUrl);
            json.put("nCtx", nCtx);
            json.put("nThreads", nThreads);
            json.put("nBatch", nBatch);
            json.put("temp", temp);
            json.put("topP", topP);
            json.put("topK", topK);
            json.put("promptTemplate", promptTemplate);
            
            // Penalty parameters
            json.put("penaltyLastN", penaltyLastN);
            json.put("penaltyRepeat", penaltyRepeat);
            json.put("penaltyFreq", penaltyFreq);
            json.put("penaltyPresent", penaltyPresent);
            
            // Mirostat parameters
            json.put("mirostat", mirostat);
            json.put("mirostatTau", mirostatTau);
            json.put("mirostatEta", mirostatEta);
            
            // Additional sampling parameters
            json.put("minP", minP);
            json.put("typicalP", typicalP);
            json.put("dynatempRange", dynatempRange);
            json.put("dynatempExponent", dynatempExponent);
            json.put("xtcProbability", xtcProbability);
            json.put("xtcThreshold", xtcThreshold);
            json.put("topNSigma", topNSigma);
            
            // DRY parameters
            json.put("dryMultiplier", dryMultiplier);
            json.put("dryBase", dryBase);
            json.put("dryAllowedLength", dryAllowedLength);
            json.put("dryPenaltyLastN", dryPenaltyLastN);
            json.put("drySequenceBreakers", drySequenceBreakers);
            
            return json;
        }
        
        public static Configuration fromJSON(JSONObject json) throws JSONException {
            Configuration config = new Configuration();
            config.name = json.getString("name");
            config.modelUrl = json.getString("modelUrl");
            config.nCtx = json.getInt("nCtx");
            config.nThreads = json.getInt("nThreads");
            config.nBatch = json.getInt("nBatch");
            config.temp = json.getDouble("temp");
            config.topP = json.getDouble("topP");
            config.topK = json.getInt("topK");
            config.promptTemplate = json.getString("promptTemplate");
            
            // Penalty parameters (with defaults for backward compatibility)
            config.penaltyLastN = json.optInt("penaltyLastN", 64);
            config.penaltyRepeat = json.optDouble("penaltyRepeat", 1.0);
            config.penaltyFreq = json.optDouble("penaltyFreq", 0.0);
            config.penaltyPresent = json.optDouble("penaltyPresent", 0.0);
            
            // Mirostat parameters (with defaults for backward compatibility)
            config.mirostat = json.optInt("mirostat", 0);
            config.mirostatTau = json.optDouble("mirostatTau", 5.0);
            config.mirostatEta = json.optDouble("mirostatEta", 0.1);
            
            // Additional sampling parameters (with defaults for backward compatibility)
            config.minP = json.optDouble("minP", 0.05);
            config.typicalP = json.optDouble("typicalP", 1.0);
            config.dynatempRange = json.optDouble("dynatempRange", 0.0);
            config.dynatempExponent = json.optDouble("dynatempExponent", 1.0);
            config.xtcProbability = json.optDouble("xtcProbability", 0.0);
            config.xtcThreshold = json.optDouble("xtcThreshold", 0.1);
            config.topNSigma = json.optDouble("topNSigma", -1.0);
            
            // DRY parameters (with defaults for backward compatibility)
            config.dryMultiplier = json.optDouble("dryMultiplier", 0.0);
            config.dryBase = json.optDouble("dryBase", 1.75);
            config.dryAllowedLength = json.optInt("dryAllowedLength", 2);
            config.dryPenaltyLastN = json.optInt("dryPenaltyLastN", -1);
            config.drySequenceBreakers = json.optString("drySequenceBreakers", "\\n,:,\",*");
            
            return config;
        }
    }
    
    public ConfigurationManager(Context context) {
        this.context = context;
        this.configDir = new File(context.getExternalFilesDir(null), CONFIG_DIR);
        if (!configDir.exists()) {
            configDir.mkdirs();
        }
        ensureDefaultConfig();
    }
    
    private void ensureDefaultConfig() {
        File defaultFile = new File(configDir, DEFAULT_CONFIG_NAME + ".json");
        if (!defaultFile.exists()) {
            try {
                saveConfiguration(new Configuration());
                Log.d(TAG, "Created default configuration");
            } catch (IOException | JSONException e) {
                Log.e(TAG, "Failed to create default configuration", e);
            }
        }
    }
    
    public void saveConfiguration(Configuration config) throws IOException, JSONException {
        if (config.name == null || config.name.trim().isEmpty()) {
            throw new IllegalArgumentException("Configuration name cannot be empty");
        }
        
        File configFile = new File(configDir, config.name + ".json");
        try (FileWriter writer = new FileWriter(configFile)) {
            writer.write(config.toJSON().toString(2)); // Pretty print with indent of 2
        }
        Log.d(TAG, "Saved configuration: " + config.name);
    }
    
    public Configuration loadConfiguration(String name) throws IOException, JSONException {
        File configFile = new File(configDir, name + ".json");
        if (!configFile.exists()) {
            throw new IOException("Configuration not found: " + name);
        }
        
        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        
        JSONObject json = new JSONObject(sb.toString());
        Configuration config = Configuration.fromJSON(json);
        Log.d(TAG, "Loaded configuration: " + name);
        return config;
    }
    
    public List<String> listConfigurations() {
        List<String> configs = new ArrayList<>();
        File[] files = configDir.listFiles((dir, name) -> name.endsWith(".json"));
        if (files != null) {
            for (File file : files) {
                String name = file.getName();
                // Remove .json extension
                configs.add(name.substring(0, name.length() - 5));
            }
        }
        return configs;
    }
    
    public boolean deleteConfiguration(String name) {
        if (DEFAULT_CONFIG_NAME.equals(name)) {
            Log.w(TAG, "Cannot delete default configuration");
            return false;
        }
        
        File configFile = new File(configDir, name + ".json");
        boolean deleted = configFile.delete();
        if (deleted) {
            Log.d(TAG, "Deleted configuration: " + name);
        }
        return deleted;
    }
}
