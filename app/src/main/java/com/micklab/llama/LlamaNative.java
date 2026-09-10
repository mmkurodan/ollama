package com.micklab.llama;

import android.util.Log;

public class LlamaNative {

    private static final String TAG = "LlamaNative";
    
    public interface DownloadProgressListener {
        void onProgress(int percent);
    }
    
    private volatile DownloadProgressListener downloadProgressListener;

    // Token listener for streaming
    public interface TokenListener {
        void onToken(String token);
        void onComplete();
        void onError(String error);
    }

    static {
        System.loadLibrary("llama_jni");
    }

    public native String download(String url, String path);
    public native void setDownloadCaBundlePath(String path);
    public native String init(String modelPath);
    public native String initWithMmproj(String modelPath, String mmprojPath, boolean enableAudio);

    /**
     * Cheaply check (GGUF metadata only, without building the model) whether {@code mmprojPath}
     * is a usable multimodal projector for {@code modelPath}. Returns one of:
     * <ul>
     *   <li>{@code "ok"} — confidently compatible</li>
     *   <li>{@code "incompatible:..."} — confidently incompatible (e.g. not an mmproj, or the
     *       projector/text-model embedding dimensions differ); loading it would likely crash</li>
     *   <li>{@code "unknown"} — could not determine (caller should fall back to other checks)</li>
     * </ul>
     * Never throws across the JNI boundary.
     */
    public native String validateMmproj(String modelPath, String mmprojPath);
    /** 1 = mmproj has an audio encoder, 0 = vision-only, -1 = unknown (couldn't read). */
    public native int mmprojSupportsAudio(String mmprojPath);
    public native void setLoadParameters(int nCtx, int nThreads, int nBatch, float temp, float topP, int topK, int nGpuLayers);
    /**
     * Configure MTP (multi-token prediction) speculative decoding. Applied at the next
     * {@link #init}/{@link #initWithMmproj}. Pass {@code enabled=false} or an empty path to
     * use plain decoding. {@code mtpModelPath} is the MTP-head draft GGUF (a *mtp* sidecar
     * of the main model); {@code nDraft} is the max tokens drafted per step (e.g. 4).
     */
    public native void setSpeculative(String mtpModelPath, int nDraft, boolean enabled);
    public native String generate(String prompt);
    public native String generateWithMedia(String prompt, byte[][] mediaFiles);
    public native String generateOpenAiChatCompletion(
        String messagesJson,
        String toolsJson,
        String chatTemplateOverride,
        String toolChoiceJson,
        boolean parallelToolCalls,
        boolean enableThinking,
        byte[][] mediaFiles
    );
    public native void cancelGeneration();
    public native void free();

    // Structured output: set the GBNF grammar for the next generate() call.
    // Pass a raw GBNF string in `gbnf`, or a JSON Schema string in `jsonSchema`
    // (converted natively via json_schema_to_grammar); pass "" for both to clear.
    public native void setGrammar(String gbnf, String jsonSchema);

    // Embedding generation: returns JSON {"embedding":[...]} or {"error":"..."}.
    // Mean-pools per-token hidden states for generative models; best with an embedding model.
    public native String embed(String text);

    // Tokenize text; returns JSON {"tokens":[...],"count":N,"ids":[...]} or {"error":"..."}.
    public native String tokenize(String text);

    // Set max output tokens per generation (-1 = use remaining context).
    public native void setNPredict(int n);

    // Set KV cache quantization types (GGML type IDs; 1=F16, 8=Q8_0, 7=Q5_1, 6=Q5_0, 3=Q4_1, 2=Q4_0, 20=IQ4_NL).
    public native void setKvCacheType(int typeK, int typeV);

    // Enable/disable memory-mapping the model file at load (default true). When false, weights
    // are read into native memory, avoiding the large contiguous virtual-address reservation
    // that mmap needs. Applied at the next model init/initWithMmproj.
    public native void setUseMmap(boolean useMmap);

    // LoRA アダプタ（別ファイルのアダプタGGUF）を現在のモデル/コンテキストへ適用する。
    // 戻り値: 成功="", 失敗=エラーメッセージ。scale は実効強度（通常 1.0）。
    public native String applyLoraAdapter(String adapterPath, float scale);
    // 適用中の LoRA アダプタを外して解放する。未適用なら何もしない。
    public native void clearLoraAdapter();
    // 適用中アダプタ情報 "path\tscale"（未適用は ""）。UI 表示用。
    public native String getLoraAdapterInfo();

    // Token listener registration (native will keep a global ref)
    public native void setTokenListener(TokenListener listener);

    // 新しく追加したネイティブ: JNI 側のログファイルパスを設定する
    public native void setLogPath(String path);
    public native void setLogLevel(int level);

    /**
     * Compute backend を設定する。
     *
     * @param backendType  0=CPU, 1=GPU
     * @param npuEnabled   常に false (NPU 対応は削除済み; シグネチャは後方互換のため保持)
     * @param nativeLibDir context.getApplicationInfo().nativeLibraryDir
     */
    public native void setBackendConfig(int backendType, boolean npuEnabled, String nativeLibDir);
    
    // Set sampler chain order (semicolon-delimited names, e.g. "top_k;top_p;temperature").
    // Empty string resets to the built-in default order.
    public native void setSamplerOrder(String order);

    // Set sampling parameters
    public native void setParameters(
        int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent,
        int mirostat, float mirostatTau, float mirostatEta,
        float minP, float typicalP,
        float dynatempRange, float dynatempExponent,
        float xtcProbability, float xtcThreshold,
        float topNSigma,
        float dryMultiplier, float dryBase, int dryAllowedLength, int dryPenaltyLastN,
        String drySequenceBreakers
    );
    
    // Get chat template from loaded GGUF model metadata
    public native String getChatTemplate();
    public native boolean supportsVision();
    public native boolean supportsAudio();

    /** Decode throughput (tokens/sec) of the most recent generation, 0 if none yet. */
    public native double getLastGenerationSpeed();

    /** Input (prompt) token count of the most recent generation. */
    public native int getLastNPromptTokens();

    /** Output (eval) token count of the most recent generation. */
    public native int getLastNEvalTokens();

    /**
     * KV-cache bytes per cell (= per token) for the currently loaded model and KV quant types,
     * or 0 if no model is loaded. The KV size at context length N is roughly
     * {@code getKvBytesPerCell() * N}. Used to bound n_ctx auto-promotion against a memory budget.
     */
    public native long getKvBytesPerCell();

    /** Total generation time (prompt + decode) in milliseconds of the most recent generation. */
    public native double getLastTotalTimeMs();

    /** Prompt processing (prefill) time in milliseconds of the most recent generation. */
    public native double getLastPromptEvalTimeMs();

    /** Model load time in milliseconds of the most recent model load. */
    public native double getLastLoadTimeMs();

    /** Effective compute backend after the last model load (CPU/GPU/CPU (fallback)). */
    public native String getActiveBackend();

    public void setDownloadProgressListener(DownloadProgressListener listener) {
        this.downloadProgressListener = listener;
    }
    
    // Called from native code to deliver download progress (0-100)
    public void onDownloadProgress(int percent) {
        Log.d(TAG, "Download progress: " + percent + "%");
        DownloadProgressListener listener = downloadProgressListener;
        if (listener != null) {
            listener.onProgress(percent);
        }
    }
}
