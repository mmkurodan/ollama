#include <jni.h>
#include <algorithm>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <cerrno>
#include <cstring>
#include <exception>
#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#if defined(__ANDROID__)
#include <dlfcn.h>
#include <malloc.h>
#endif

#include <android/log.h>
#define LOG_TAG "LLAMA_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

#include <nlohmann/json.hpp>
#include <cstdio>

#include "llama.h"
#include "ggml.h"          // ggml_type_size / ggml_blck_size (KV bytes/cell computation)
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"   // ★ これが必要
#include "ggml-cpu.h"
#if defined(GGML_USE_HEXAGON)
#include "ggml-hexagon.h"
#endif
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"
#include "common.h"
#include "speculative.h"
#include "json-schema-to-grammar.h"
#include <curl/curl.h>

using json = nlohmann::ordered_json;

// ---------------- グローバル ----------------
static std::mutex g_mutex;
static llama_model   *g_model = nullptr;
static llama_context *g_ctx   = nullptr;
// 適用中の LoRA アダプタ（g_model に紐づく。model 解放時に自動 free されるため g_lora も無効化する）。
static llama_adapter_lora *g_lora       = nullptr;
static std::string         g_lora_path;
static float               g_lora_scale = 1.0f;
// Grammar constraint applied via common_sampler in generate(); empty (type NONE) = no constraint.
// Set via setGrammar() before a generate() call (OllamaApiServer wires format/grammar here).
// USER = raw GBNF; OUTPUT_FORMAT = JSON schema (converted by common_sampler).
static common_grammar g_grammar;
static mtmd_context  *g_mtmd  = nullptr;

// common_context_seq_rm shim.
// Up to b10091 llama.cpp exported common_context_seq_rm() from common.h. The
// b10621/v0.3.0 KV-memory refactor made it a file-static helper inside common.cpp
// (now reached via common_memory::seq_rm), so it is no longer linkable from this
// translation unit. Reintroduce a local equivalent over the public llama_memory_seq_rm()
// API to keep the MTP draft/target KV trimming working. Unlike upstream's
// GGML_ABORT-on-failure variant we only log, so a recoverable partial-removal failure
// never turns into a process abort on this memory-sensitive app.
static void common_context_seq_rm(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (ctx == nullptr) {
        return;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem != nullptr && !llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        LOGE("common_context_seq_rm: partial removal failed (seq=%d p0=%d p1=%d)", seq_id, p0, p1);
    }
}

// ---- MTP / speculative decoding (draft-mtp) ----
// g_spec_init owns the draft (MTP-head) model + its context; g_ctx_dft is borrowed
// from it. g_spec_cp must outlive g_spec because common_speculative_init() takes its
// speculative params by reference. All are set up in maybe_init_speculative_locked()
// after g_ctx exists, and torn down in release_speculative_locked() (before g_ctx).
static common_speculative              *g_spec = nullptr;
static common_speculative_init_result_ptr g_spec_init;   // owns model_dft + ctx_dft
static llama_context                   *g_ctx_dft = nullptr; // borrowed from g_spec_init
static common_params                    g_spec_cp;       // kept alive for g_spec
static common_context_seq_rm_type       g_ctx_tgt_rm = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
static common_context_seq_rm_type       g_ctx_dft_rm = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
static llama_batch                      g_spec_batch = {};   // [id_last, draft...] eval batch
static std::vector<llama_token>         g_spec_prompt;       // committed tokens (draft params .prompt)
// Per-generation MTP stats (reset before the loop, logged after). g_spec_stat_eval counts
// every position fed to llama_decode during speculation (incl. rejected drafts + retries);
// on a compute-bound CPU each position ~= one token of work, so eval/out is the slowdown
// factor vs. plain decoding (>1 means speculation is costing more than it produces).
static long g_spec_stat_steps = 0, g_spec_stat_out = 0, g_spec_stat_eval = 0;
// Config pushed from Java via setSpeculative(); applied at the next init/initWithMmproj.
static std::string g_mtp_path;            // path to the MTP-head draft GGUF ("" = none)
static int32_t     g_mtp_n_draft = 2;     // max tokens drafted per step (MTP heads reliably predict ~1-2 ahead)
static bool        g_mtp_enabled = false; // master toggle (default off = unchanged behaviour)

static JavaVM *g_jvm = nullptr;
// Token listener global ref and method IDs
static jobject g_token_listener = nullptr;
static jmethodID g_token_onToken = nullptr;
static jmethodID g_token_onComplete = nullptr;
static jmethodID g_token_onError = nullptr;
static std::atomic<bool> g_cancel_generation(false);
static bool g_backend_initialized = false;

// ---- Compute backend configuration ----
// backendType: 0=CPU, 1=GPU, 2=NPU/HTP, 3=GPU+NPU
static int         g_backend_type          = 0;
static bool        g_npu_enabled           = false;
static std::string g_native_lib_dir;
// Version counter — bumped in setBackendConfig to trigger re-init on next model load
static int         g_backend_config_version      = 0;
static int         g_backend_initialized_version = -1;
// Persistent NULL-terminated device list for llama_model_params.devices
static ggml_backend_dev_t g_accel_devices[4] = {nullptr, nullptr, nullptr, nullptr};

// Effective compute backend after the last model build (for UI display).
// One of: "CPU", "GPU", "NPU/HTP", "GPU+NPU", "CPU (fallback)".
// Atomic (lock-free) so the UI thread can read it via getActiveBackend() WITHOUT taking
// g_mutex — generate*() hold g_mutex for the whole inference, and blocking the UI poller on
// it caused freezes/ANR during long (VL/MCP) runs. Values are static string literals.
static std::atomic<const char *> g_active_backend{"CPU"};

// Keep track of currently loaded model path to avoid redundant inits
static std::string g_current_model_path;
static std::string g_current_mmproj_path;
static bool g_supports_vision = false;
static bool g_supports_audio = false;
static bool g_current_audio_requested = false;
static std::mutex g_download_ca_bundle_mutex;
static std::string g_download_ca_bundle_path;

// ログ用
static std::mutex g_log_mutex;
static std::string g_log_path;
static std::ofstream g_log_ofs;
static std::atomic<int> g_log_level(GGML_LOG_LEVEL_INFO);
static volatile sig_atomic_t g_signal_log_fd = -1;
static volatile sig_atomic_t g_signal_crash_fd = -1;
static std::atomic<bool> g_fatal_signal_handlers_installed(false);
static constexpr const char * NATIVE_CRASH_LOG_FILENAME = "native_crash.txt";
static void log_to_file(const std::string& msg, ggml_log_level level = GGML_LOG_LEVEL_INFO);
static void trim_native_allocator(const char * reason);
static std::string initialize_optional_multimodal_support_locked(
        const std::string & model_path,
        const std::string & requested_mmproj_path,
        bool enable_audio,
        const char * log_prefix);

// Prefix KV-cache reuse (MCP / multi-turn): the exact tokens currently held in the KV
// cache (seq 0). Reused across OpenAI-path text generations so the shared conversation
// prefix is not re-prefilled every tool turn. Cleared whenever the KV is reset/freed.
static std::vector<llama_token> g_kv_cached_tokens;

// 設定
static int   g_n_ctx      = 2048;
static int   g_n_threads  = 2;
static int   g_n_batch    = 16;
static float g_temp       = 0.7f;
static float g_top_p      = 0.9f;
static int   g_top_k      = 40;
static int   g_n_gpu_layers = 0;
// Memory-map the model file at load (default true). When false, weights are read into
// native memory, avoiding the single large contiguous virtual-address reservation mmap
// needs — helps very large models load when address space is fragmented.
static bool  g_use_mmap   = true;

// DRY sequence breakers default - MUST match Java ConfigurationManager.Configuration.DEFAULT_DRY_SEQUENCE_BREAKERS
static const char* DEFAULT_DRY_SEQUENCE_BREAKERS = "\\n,:,\",*";

// Penalty parameters
static int   g_penalty_last_n    = 64;
static float g_penalty_repeat    = 1.0f;
static float g_penalty_freq      = 0.0f;
static float g_penalty_present   = 0.0f;

// Mirostat parameters
static int   g_mirostat          = 0;
static float g_mirostat_tau      = 5.0f;
static float g_mirostat_eta      = 0.1f;

// Additional sampler parameters
static float g_min_p             = 0.05f;
static float g_typical_p         = 1.0f;
static float g_dynatemp_range    = 0.0f;
static float g_dynatemp_exponent = 1.0f;
static float g_xtc_probability   = 0.0f;
static float g_xtc_threshold     = 0.1f;
static float g_top_n_sigma       = -1.0f;

// DRY parameters
static float g_dry_multiplier       = 0.0f;
static float g_dry_base             = 1.75f;
static int   g_dry_allowed_length   = 2;
static int   g_dry_penalty_last_n   = -1;
static std::string g_dry_sequence_breakers = DEFAULT_DRY_SEQUENCE_BREAKERS;

// Sampler chain order: semicolon-delimited names (e.g. "top_k;top_p;temperature").
// Empty = use built-in default order.  Reset to "" after each generation so that
// direct-input calls that do not supply an order fall back to the default.
static std::string g_sampler_order = "";

// Generation length limit: -1 means use remaining context (n_ctx - n_past).
static int g_n_predict = -1;

// KV cache quantization types (GGML type IDs). Default 1 = GGML_TYPE_F16.
static int g_kv_type_k = 1;
static int g_kv_type_v = 1;

// Common stop sequences for chat templates to prevent runaway generation
static const std::vector<std::string> DEFAULT_STOP_SEQUENCES = {
        "<|end|>",
        "</s>",
        "<|eot_id|>",
        "<|im_end|>",
        "<|IM_END|>",
        "<|im_end|",
        "<|IM_END|",
        "<|im_end",
        "<|IM_END",
        "<end",
        "<END",
        "<end_of_turn>",
        "<|user|>",
        "<|system|>",
        "<|im_start|>user",
        "<|im_start|>system",
        "<start_of_turn>user",
        "<start_of_turn>model"
};

// ---------------- ログユーティリティ ----------------
static std::string current_time_str() {
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    struct tm tm_buf;
    localtime_r(&t, &tm_buf);
    std::ostringstream ss;
    ss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static ggml_log_level normalize_log_level(ggml_log_level level) {
    return level == GGML_LOG_LEVEL_NONE ? GGML_LOG_LEVEL_INFO : level;
}

static bool should_log(ggml_log_level level) {
    const int threshold = g_log_level.load();
    const ggml_log_level effective = normalize_log_level(level);
    return effective >= threshold;
}

static bool should_log_debug() {
    return should_log(GGML_LOG_LEVEL_DEBUG);
}

static bool is_max_debug_mode() {
    return g_log_level.load() == GGML_LOG_LEVEL_NONE;
}

static size_t append_literal(char * buffer, size_t offset, size_t capacity, const char * text) {
    if (!text || offset >= capacity) {
        return offset;
    }
    while (*text != '\0' && offset < capacity) {
        buffer[offset++] = *text++;
    }
    return offset;
}

static size_t append_decimal(char * buffer, size_t offset, size_t capacity, int value) {
    if (offset >= capacity) {
        return offset;
    }
    if (value == 0) {
        if (offset < capacity) {
            buffer[offset++] = '0';
        }
        return offset;
    }

    unsigned int magnitude;
    if (value < 0) {
        if (offset < capacity) {
            buffer[offset++] = '-';
        }
        magnitude = static_cast<unsigned int>(-(value + 1)) + 1u;
    } else {
        magnitude = static_cast<unsigned int>(value);
    }

    char digits[16];
    size_t count = 0;
    while (magnitude > 0 && count < sizeof(digits)) {
        digits[count++] = static_cast<char>('0' + (magnitude % 10u));
        magnitude /= 10u;
    }
    while (count > 0 && offset < capacity) {
        buffer[offset++] = digits[--count];
    }
    return offset;
}

static const char * fatal_signal_name(int signo) {
    switch (signo) {
        case SIGABRT: return "SIGABRT";
        case SIGBUS:  return "SIGBUS";
        case SIGFPE:  return "SIGFPE";
        case SIGILL:  return "SIGILL";
        case SIGSEGV: return "SIGSEGV";
        default:      return "UNKNOWN";
    }
}

static void write_best_effort(int fd, const char * buffer, size_t length) {
    if (fd < 0 || !buffer || length == 0) {
        return;
    }
    size_t written = 0;
    while (written < length) {
        const ssize_t rc = write(fd, buffer + written, length - written);
        if (rc > 0) {
            written += static_cast<size_t>(rc);
            continue;
        }
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        break;
    }
}

static void write_native_crash_marker(int fd, int signo) {
    char buffer[256];
    size_t len = 0;
    len = append_literal(buffer, len, sizeof(buffer), "\n=== Native fatal signal ===\nSignal: ");
    len = append_literal(buffer, len, sizeof(buffer), fatal_signal_name(signo));
    len = append_literal(buffer, len, sizeof(buffer), " (");
    len = append_decimal(buffer, len, sizeof(buffer), signo);
    len = append_literal(buffer, len, sizeof(buffer), ")\nThe process aborted before Java crash handlers could run.\nSee ollama.log for the preceding JNI/llama.cpp lines.\n");
    write_best_effort(fd, buffer, len);
}

static void native_fatal_signal_handler(int signo, siginfo_t *, void *) {
    const int saved_errno = errno;
    const int crash_fd = static_cast<int>(g_signal_crash_fd);
    const int log_fd = static_cast<int>(g_signal_log_fd);

    write_native_crash_marker(crash_fd, signo);
    if (log_fd >= 0 && log_fd != crash_fd) {
        write_native_crash_marker(log_fd, signo);
    }

    errno = saved_errno;
    kill(getpid(), signo);
    _exit(128 + signo);
}

static void ensure_fatal_signal_handlers_installed() {
    if (g_fatal_signal_handlers_installed.exchange(true)) {
        return;
    }

    struct sigaction action = {};
    sigemptyset(&action.sa_mask);
    action.sa_sigaction = native_fatal_signal_handler;
    action.sa_flags = SA_SIGINFO | SA_RESETHAND | SA_NODEFER;

    const int fatal_signals[] = {SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV};
    for (const int signo : fatal_signals) {
        sigaction(signo, &action, nullptr);
    }
}

static std::string derive_crash_log_path(const std::string & log_path) {
    if (log_path.empty()) {
        return "";
    }
    const size_t slash = log_path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return NATIVE_CRASH_LOG_FILENAME;
    }
    return log_path.substr(0, slash + 1) + NATIVE_CRASH_LOG_FILENAME;
}

static void replace_signal_fd(volatile sig_atomic_t * target, int new_fd) {
    if (!target || new_fd < 0) {
        return;
    }
    const int old_fd = static_cast<int>(*target);
    *target = static_cast<sig_atomic_t>(new_fd);
    if (old_fd >= 0 && old_fd != new_fd) {
        close(old_fd);
    }
}

// Return the last index that ends on a valid UTF-8 boundary (trims incomplete trailing bytes).
static size_t validate_utf8(const std::string& text) {
    size_t len = text.size();
    if (len == 0) return 0;

    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        unsigned char c = static_cast<unsigned char>(text[len - i]);
        if ((c & 0xE0) == 0xC0) {
            if (i < 2) return len - i;
            break;
        } else if ((c & 0xF0) == 0xE0) {
            if (i < 3) return len - i;
            break;
        } else if ((c & 0xF8) == 0xF0) {
            if (i < 4) return len - i;
            break;
        } else if ((c & 0x80) == 0x00) {
            break;
        }
    }

    return len;
}

static void append_replacement_char(std::u16string & out) {
    out.push_back(static_cast<char16_t>(0xFFFD));
}

static void append_code_point_utf16(std::u16string & out, uint32_t code_point) {
    if (code_point <= 0xFFFF) {
        out.push_back(static_cast<char16_t>(code_point));
        return;
    }

    code_point -= 0x10000;
    out.push_back(static_cast<char16_t>(0xD800 + ((code_point >> 10) & 0x3FF)));
    out.push_back(static_cast<char16_t>(0xDC00 + (code_point & 0x3FF)));
}

static std::u16string utf8_to_utf16_lossy(const std::string & text) {
    std::u16string out;
    out.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c0 = static_cast<unsigned char>(text[i]);
        uint32_t code_point = 0;
        size_t width = 0;
        bool valid = false;

        if (c0 <= 0x7F) {
            code_point = c0;
            width = 1;
            valid = true;
        } else if (c0 >= 0xC2 && c0 <= 0xDF && i + 1 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            if ((c1 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
                width = 2;
                valid = true;
            }
        } else if (c0 == 0xE0 && i + 2 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            if (c1 >= 0xA0 && c1 <= 0xBF && (c2 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
                width = 3;
                valid = true;
            }
        } else if ((c0 >= 0xE1 && c0 <= 0xEC || c0 >= 0xEE && c0 <= 0xEF) && i + 2 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
                width = 3;
                valid = true;
            }
        } else if (c0 == 0xED && i + 2 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            if (c1 >= 0x80 && c1 <= 0x9F && (c2 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
                width = 3;
                valid = true;
            }
        } else if (c0 == 0xF0 && i + 3 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            const unsigned char c3 = static_cast<unsigned char>(text[i + 3]);
            if (c1 >= 0x90 && c1 <= 0xBF
                    && (c2 & 0xC0) == 0x80
                    && (c3 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x07) << 18)
                        | ((c1 & 0x3F) << 12)
                        | ((c2 & 0x3F) << 6)
                        | (c3 & 0x3F);
                width = 4;
                valid = true;
            }
        } else if ((c0 >= 0xF1 && c0 <= 0xF3) && i + 3 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            const unsigned char c3 = static_cast<unsigned char>(text[i + 3]);
            if ((c1 & 0xC0) == 0x80
                    && (c2 & 0xC0) == 0x80
                    && (c3 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x07) << 18)
                        | ((c1 & 0x3F) << 12)
                        | ((c2 & 0x3F) << 6)
                        | (c3 & 0x3F);
                width = 4;
                valid = true;
            }
        } else if (c0 == 0xF4 && i + 3 < text.size()) {
            const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
            const unsigned char c3 = static_cast<unsigned char>(text[i + 3]);
            if (c1 >= 0x80 && c1 <= 0x8F
                    && (c2 & 0xC0) == 0x80
                    && (c3 & 0xC0) == 0x80) {
                code_point = ((c0 & 0x07) << 18)
                        | ((c1 & 0x3F) << 12)
                        | ((c2 & 0x3F) << 6)
                        | (c3 & 0x3F);
                width = 4;
                valid = true;
            }
        }

        if (!valid) {
            append_replacement_char(out);
            ++i;
            continue;
        }

        append_code_point_utf16(out, code_point);
        i += width;
    }

    return out;
}

static jstring new_java_string_utf8(JNIEnv * env, const std::string & text) {
    static const jchar empty_chars[1] = {0};
    const std::u16string utf16 = utf8_to_utf16_lossy(text);
    const jchar * chars = utf16.empty()
            ? empty_chars
            : reinterpret_cast<const jchar *>(utf16.data());
    return env->NewString(chars, static_cast<jsize>(utf16.size()));
}

static jstring new_java_string_utf8(JNIEnv * env, const char * text) {
    return new_java_string_utf8(env, text != nullptr ? std::string(text) : std::string());
}

static bool find_stop_sequence(const std::string& text, size_t * stop_pos, std::string * stop_seq) {
    size_t earliest = std::string::npos;
    std::string found;
    for (const auto& seq : DEFAULT_STOP_SEQUENCES) {
        size_t pos = text.find(seq);
        if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
            earliest = pos;
            found = seq;
        }
    }
    if (earliest != std::string::npos) {
        if (stop_pos) *stop_pos = earliest;
        if (stop_seq) *stop_seq = found;
        return true;
    }
    return false;
}

static void log_requested_load_params(const char * log_prefix) {
    std::ostringstream ss;
    ss << log_prefix
       << ": requested load params n_ctx=" << g_n_ctx
       << " n_threads=" << g_n_threads
       << " n_batch=" << g_n_batch
       << " n_gpu_layers=" << g_n_gpu_layers
       << " temp=" << g_temp
       << " top_p=" << g_top_p
       << " top_k=" << g_top_k;
    log_to_file(ss.str());
}

static std::string build_model_load_failure_message(const char * log_prefix) {
    std::ostringstream ss;
    ss << log_prefix << ": failed to load model";
    if (g_use_mmap) {
        // mmap is on: a large contiguous virtual-address reservation is required, so
        // address-space fragmentation is a plausible cause and disabling mmap may help.
        ss << " (out of memory, or mmap address-space fragmentation — for very large models"
              " try disabling memory-map / lowering n_gpu_layers / n_ctx)";
    } else {
        // mmap is off: weights are read into native memory, so this is not an mmap issue.
        ss << " (out of memory, or the model file is missing/corrupt — this is not an mmap issue;"
              " try lowering n_gpu_layers / n_ctx or use a smaller model)";
    }
    return ss.str();
}

static std::string build_context_load_failure_message(const char * log_prefix) {
    std::ostringstream ss;
    ss << log_prefix
       << ": failed to create context (n_ctx=" << g_n_ctx
       << ", possible commit/address-space exhaustion)";
    return ss.str();
}

static std::string summarize_registered_backends() {
    const size_t backend_count = ggml_backend_reg_count();
    std::ostringstream ss;
    ss << "registered backends=" << backend_count;
    if (backend_count > 0) {
        ss << " [";
        for (size_t i = 0; i < backend_count; ++i) {
            if (i > 0) {
                ss << ", ";
            }
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            ss << (reg ? ggml_backend_reg_name(reg) : "<null>");
        }
        ss << "]";
    }
    return ss.str();
}

static int32_t detokenize_with_resize(
        const llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
        std::string & out_text) {
    if (tokens.empty()) {
        out_text.clear();
        return 0;
    }

    size_t target = tokens.size();
    if (out_text.capacity() > target) {
        target = out_text.capacity();
    }
    out_text.resize(target);

    int32_t n_chars = llama_detokenize(
            vocab,
            tokens.data(),
            (int32_t)tokens.size(),
            &out_text[0],
            (int32_t)out_text.size(),
            true,
            false
    );

    if (n_chars < 0) {
        out_text.resize(-n_chars);
        n_chars = llama_detokenize(
                vocab,
                tokens.data(),
                (int32_t)tokens.size(),
                &out_text[0],
                (int32_t)out_text.size(),
                true,
                false
        );
    }

    if (n_chars < 0) {
        return n_chars;
    }

    out_text.resize(n_chars);
    return n_chars;
}

struct reusable_token_batch {
    llama_batch batch;

    explicit reusable_token_batch(int32_t capacity)
            : batch(llama_batch_init(std::max<int32_t>(capacity, 1), 0, 1)) {
    }

    ~reusable_token_batch() {
        llama_batch_free(batch);
    }

    bool valid() const {
        return batch.token != nullptr
                && batch.pos != nullptr
                && batch.n_seq_id != nullptr
                && batch.seq_id != nullptr
                && batch.seq_id[0] != nullptr
                && batch.logits != nullptr;
    }

    void set_token(int32_t index, llama_token token, llama_pos pos, bool logits) {
        batch.token[index] = token;
        batch.pos[index] = pos;
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = logits ? 1 : 0;
    }
};

static llama_context_params build_context_params_locked() {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = g_n_ctx;
    cparams.n_threads       = g_n_threads;
    cparams.n_batch         = g_n_batch;
    cparams.n_ubatch        = g_n_batch;
    cparams.n_threads_batch = g_n_threads;
    cparams.no_perf         = false;  // 計測を有効化 (既定 true だと t_eval=0 で tok/s 不能)
    cparams.type_k          = static_cast<ggml_type>(g_kv_type_k);
    cparams.type_v          = static_cast<ggml_type>(g_kv_type_v);
    return cparams;
}

// Return the first registered Hexagon/HTP device, or nullptr if none found.
static ggml_backend_dev_t find_hexagon_device_locked() {
#if defined(GGML_USE_HEXAGON)
    const size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) { continue; }
        const char * name = ggml_backend_dev_name(dev);
        if (!name) { continue; }
        const std::string n(name);
        if (n.find("Hexagon") != std::string::npos || n.find("HTP") != std::string::npos) {
            return dev;
        }
    }
#endif
    return nullptr;
}

// Return the first registered GPU device, or nullptr if none found.
static ggml_backend_dev_t find_gpu_device_locked() {
    const size_t count = ggml_backend_dev_count();
    // Prefer a real GPU (OpenCL/Adreno). The Hexagon device also reports type GPU,
    // so exclude it here — NPU is selected separately via find_hexagon_device_locked.
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) { continue; }
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        const char * name = ggml_backend_dev_name(dev);
        const std::string n = name ? name : "";
        if (n.find("Hexagon") != std::string::npos || n.find("HTP") != std::string::npos) {
            continue;
        }
        return dev;
    }
    return nullptr;
}

static llama_model_params build_model_params_locked(bool has_explicit_mmproj) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = g_n_gpu_layers;
    (void) has_explicit_mmproj;
    // b10621/v0.3.0 replaced the use_mmap/use_mlock booleans with a load_mode enum.
    // We never used mlock, so map the app's mmap toggle onto MMAP vs NONE.
    mparams.load_mode = g_use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;

    // ---- Backend device list ----
    // backendType: 0=CPU  1=GPU  2=NPU/HTP  3=GPU+NPU
    // npuEnabled : quick toggle; when false, NPU portion is skipped.
    const bool want_npu = g_npu_enabled && (g_backend_type == 2 || g_backend_type == 3);
    const bool want_gpu = (g_backend_type == 1 || g_backend_type == 3);

    // GPU+NPU hybrid deadlocks in ggml's heterogeneous scheduler (Hexagon NPU + Adreno
    // OpenCL): cross-device tensor placement (e.g. a layer on GPU but its Flash-Attention
    // tensor on the NPU, or HTP-unsupported ops) stalls the very first decode. Run on a
    // SINGLE accelerator instead — prefer NPU, fall back to GPU if NPU is unavailable.
    const bool hybrid_requested = want_npu && want_gpu;
    if (hybrid_requested) {
        log_to_file("build_model_params: GPU+NPU hybrid is unstable (heterogeneous scheduler can "
                    "hang) — using a single accelerator (NPU preferred)", GGML_LOG_LEVEL_WARN);
    }

    if (want_npu || want_gpu) {
        size_t slot = 0;
        bool npu_added = false;
        bool gpu_added = false;

        if (want_npu) {
            ggml_backend_dev_t npu = find_hexagon_device_locked();
            if (npu) {
                g_accel_devices[slot++] = npu;
                npu_added = true;
                log_to_file(std::string("build_model_params: NPU/HTP device added: ") +
                            ggml_backend_dev_name(npu));
            } else {
                log_to_file("build_model_params: NPU/HTP requested but this chip has no "
                            "supported Hexagon device", GGML_LOG_LEVEL_WARN);
            }
        }

        // In hybrid mode use exactly one accelerator: only add the GPU if the NPU was not added.
        if (want_gpu && !(hybrid_requested && npu_added)) {
            ggml_backend_dev_t gpu = find_gpu_device_locked();
            if (gpu) {
                g_accel_devices[slot++] = gpu;
                gpu_added = true;
                log_to_file(std::string("build_model_params: GPU device added: ") +
                            ggml_backend_dev_name(gpu));
            } else {
                log_to_file("build_model_params: GPU requested but no GPU device found — "
                            "falling back", GGML_LOG_LEVEL_WARN);
            }
        }

        g_accel_devices[slot] = nullptr;  // NULL terminator

        if (slot > 0) {
            mparams.devices = g_accel_devices;
            g_active_backend = (npu_added && gpu_added) ? "GPU+NPU"
                             : npu_added ? "NPU/HTP"
                             : "GPU";
        } else {
            // 要求したアクセラレータ (NPU/GPU) がこのチップで利用不可 → CPU にフォールバック。
            // g_accel_devices[0] は既に nullptr。空の (NULL終端) デバイスリストを明示して全レイヤを
            // CPU に載せる。devices=NULL のままだと llama が自動登録された GPU 型デバイスを層配置に
            // 拾う (GGML_USE_HEXAGON では HTP も GPU 型として常に自動登録される) ため、DSP が健全でも
            // CPU モードが純 CPU にならない。空リストで純 CPU の層配置を保証する。
            // 注: stuck DSP (0x80000406) 由来の load_tensors 中 SIGABRT を防ぐのはこの空リストではなく、
            // ggml-backend-reg.cpp register_device の null デバイスガード (失敗 HTP セッションが
            // グローバルレジストリに null を残すのを防ぐ) である。
            mparams.devices = g_accel_devices;  // empty list → pure CPU layer placement
            mparams.n_gpu_layers = 0;
            g_active_backend = "CPU (fallback)";
            log_to_file("build_model_params: requested accelerator unavailable on this chip "
                        "— falling back to CPU (n_gpu_layers=0)", GGML_LOG_LEVEL_WARN);
        }
    } else {
        // CPU only (backendType=0)。空の (NULL終端) デバイスリストを明示する。
        // GGML_USE_HEXAGON ビルドでは HTP が ggml レジストリ初期化時に常に自動登録され、しかも
        // GPU 型として報告される。devices=NULL のままだと llama がその HTP を層配置に拾うため、
        // 空リストで純 CPU を保証する。
        // 注: stuck DSP (0x80000406) で HTP セッションが開けないとき、失敗デバイスが null として
        // グローバルレジストリに入り、load_tensors 中の make_cpu_buft_list の全デバイス列挙で
        // SIGABRT していた。これを止めるのは ggml-backend-reg.cpp register_device の null ガードで
        // あり、空リスト (model->devices) はグローバル列挙と無関係なので関与しない。
        g_accel_devices[0] = nullptr;
        mparams.devices = g_accel_devices;
        mparams.n_gpu_layers = 0;
        g_active_backend = "CPU";
    }

    return mparams;
}

static void log_model_params(const char * log_prefix, const llama_model_params & mparams) {
    static const char * const BACKEND_NAMES[] = {"CPU", "GPU", "NPU/HTP", "GPU+NPU"};
    const char * type_name = (g_backend_type >= 0 && g_backend_type < 4)
                           ? BACKEND_NAMES[g_backend_type] : "UNKNOWN";
    std::ostringstream ss;
    ss << log_prefix
       << ": model params n_gpu_layers=" << mparams.n_gpu_layers
       << " use_mmap=" << ((mparams.load_mode == LLAMA_LOAD_MODE_MMAP
                            || mparams.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK) ? "true" : "false")
       << " load_mode=" << (int) mparams.load_mode
       << " backend=" << type_name
       << " npuEnabled=" << (g_npu_enabled ? "true" : "false")
       << " devices=" << (!mparams.devices ? "default(NULL)"
                          : mparams.devices[0] ? "custom" : "none(CPU-only,HTP excluded)");
    log_to_file(ss.str());
}

static void release_multimodal_locked(const char * log_prefix) {
    if (!g_mtmd) {
        return;
    }
    mtmd_free(g_mtmd);
    g_mtmd = nullptr;
    g_current_mmproj_path.clear();
    g_supports_vision = false;
    g_supports_audio = false;
    if (log_prefix != nullptr) {
        log_to_file(std::string(log_prefix) + ": multimodal context freed");
    }
}

// Frees the speculative (draft-mtp) context. Must run before g_ctx is freed because
// g_spec holds g_ctx as its target context. Safe to call when nothing is set up.
static void release_speculative_locked(const char * log_prefix) {
    if (!g_spec && !g_spec_init) {
        return;
    }
    if (g_spec) {
        common_speculative_free(g_spec);
        g_spec = nullptr;
    }
    g_spec_init.reset();   // frees the draft model + ctx_dft
    g_ctx_dft = nullptr;
    g_ctx_tgt_rm = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    g_ctx_dft_rm = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    if (g_spec_batch.token != nullptr) {
        llama_batch_free(g_spec_batch);
        g_spec_batch = {};
    }
    g_spec_prompt.clear();
    if (log_prefix != nullptr) {
        log_to_file(std::string(log_prefix) + ": speculative context freed");
    }
}

static void release_context_locked(const char * log_prefix) {
    // The speculative context borrows g_ctx as its target; tear it down first.
    release_speculative_locked(log_prefix);
    if (!g_ctx) {
        return;
    }
    llama_synchronize(g_ctx);
    llama_free(g_ctx);
    g_ctx = nullptr;
    g_kv_cached_tokens.clear();   // context (and its KV) gone -> prefix cache invalid
    if (log_prefix != nullptr) {
        log_to_file(std::string(log_prefix) + ": context freed");
    }
}

static void release_model_locked(const char * log_prefix) {
    if (!g_model) {
        return;
    }
    llama_model_free(g_model);
    g_model = nullptr;
    g_current_model_path.clear();
    // アダプタは model に紐づき、model_free で自動解放される。ハンドルを無効化するだけ。
    g_lora = nullptr;
    g_lora_path.clear();
    if (log_prefix != nullptr) {
        log_to_file(std::string(log_prefix) + ": model freed");
    }
}

static void release_backend_locked(const char * log_prefix) {
    if (!g_backend_initialized) {
        return;
    }
    llama_backend_free();
    g_backend_initialized = false;
    if (log_prefix != nullptr) {
        log_to_file(std::string(log_prefix) + ": backend freed");
    }
}

// Sets up the MTP (draft-mtp) speculative context on top of the already-created
// g_model / g_ctx, when enabled via setSpeculative(). Non-fatal: on any failure it
// logs, tears down partial state, and leaves g_spec == nullptr so generation simply
// falls back to the normal (non-speculative) path. Call after g_ctx is created.
static void maybe_init_speculative_locked(const char * log_prefix) {
    // Start from a clean slate so a stale spec from a previous model never leaks.
    release_speculative_locked(log_prefix);

    if (!g_mtp_enabled || !g_model || !g_ctx) {
        return;
    }

    try {
        g_spec_cp = common_params{};
        g_spec_cp.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        g_spec_cp.speculative.draft.n_max = g_mtp_n_draft > 0 ? g_mtp_n_draft : 2;

        // Set a separate draft model ONLY when the user picked a different sidecar file.
        // An empty path (or the loaded model itself) means "use this model's own embedded
        // MTP head": common_speculative reuses the target model (memory-shared, no second
        // model load) -- the lightweight/correct path for Qwen3.5-MTP / DeepSeek etc. whose
        // MTP head ships inside the main GGUF (there is no separate mtp* sidecar file).
        const bool separate_draft = !g_mtp_path.empty() && g_mtp_path != g_current_model_path;
        if (separate_draft) {
            g_spec_cp.speculative.draft.mparams.path = g_mtp_path;
        }

        // Creates the MTP draft context (loads a separate draft model only if set above).
        common_params params_dft = common_base_params_to_speculative(g_spec_cp);
        g_spec_init = common_speculative_init_from_params(params_dft, g_model, g_ctx);
        g_ctx_dft = g_spec_init ? g_spec_init->context() : nullptr;
        if (!g_ctx_dft) {
            log_to_file(std::string(log_prefix) + ": MTP draft context creation failed; falling back to text-only decode",
                        GGML_LOG_LEVEL_WARN);
            release_speculative_locked(log_prefix);
            return;
        }

        g_spec_cp.speculative.draft.ctx_tgt = g_ctx;
        g_spec_cp.speculative.draft.ctx_dft = g_ctx_dft;

        g_spec = common_speculative_init(g_spec_cp.speculative, 1);
        if (!g_spec) {
            log_to_file(std::string(log_prefix) + ": common_speculative_init returned null; falling back",
                        GGML_LOG_LEVEL_WARN);
            release_speculative_locked(log_prefix);
            return;
        }

        g_ctx_tgt_rm = common_context_can_seq_rm(g_ctx);
        g_ctx_dft_rm = common_context_can_seq_rm(g_ctx_dft);

        // Eval batch for [id_last, draft0..draftN-1] (capacity n_max + 1), single sequence.
        g_spec_batch = llama_batch_init(g_spec_cp.speculative.draft.n_max + 1, 0, 1);

        std::ostringstream ss;
        ss << log_prefix << ": MTP speculative decoding enabled (draft="
           << (separate_draft ? g_mtp_path : std::string("<own embedded MTP head>"))
           << " n_draft=" << g_spec_cp.speculative.draft.n_max
           << " tgt_seq_rm=" << (int) g_ctx_tgt_rm << " dft_seq_rm=" << (int) g_ctx_dft_rm << ")";
        log_to_file(ss.str());
    } catch (const std::exception & e) {
        log_to_file(std::string(log_prefix) + ": MTP setup exception: " + e.what(), GGML_LOG_LEVEL_ERROR);
        release_speculative_locked(log_prefix);
    } catch (...) {
        log_to_file(std::string(log_prefix) + ": MTP setup unknown exception", GGML_LOG_LEVEL_ERROR);
        release_speculative_locked(log_prefix);
    }
}

// One draft-mtp speculative step over sequence 0. Drafts from id_last, evaluates
// [id_last, draft...] on g_ctx, runs the MTP process() hook, verifies with the target
// sampler and accepts the matching prefix (+1 bonus). Rolls back the rejected-draft KV:
// a direct seq_rm for PART contexts, or checkpoint-restore + re-evaluate-accepted for
// FULL contexts (hybrid/GDN such as Qwen3.5). On success out_ids holds the accepted
// tokens (>=1) and n_past/id_last are advanced. Returns false only on a hard decode
// error, so the caller can fall back to plain decoding.
static bool spec_decode_step_locked(
        common_sampler_ptr & smpl,
        llama_token & id_last,
        llama_pos & n_past,
        common_prompt_checkpoint & ckpt,
        std::vector<llama_token> & out_ids) {
    out_ids.clear();
    if (!g_spec || !smpl || g_spec_batch.token == nullptr) {
        return false;
    }

    const bool use_ckpt = (g_ctx_tgt_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    std::vector<llama_token> draft;
    bool reuse_draft = false;
    const int max_retries = 16;

    for (int attempt = 0; ; ++attempt) {
        if (!reuse_draft) {
            draft.clear();
            common_speculative_get_draft_params(g_spec, 0) = {
                /* .drafting = */ true,
                /* .n_max    = */ -1,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &g_spec_prompt,
                /* .result   = */ &draft,
            };
            common_speculative_draft(g_spec);
        }
        reuse_draft = false;

        // Checkpoint before decode when the context cannot drop a partial tail (FULL).
        if (use_ckpt) {
            llama_memory_t mem = llama_get_memory(g_ctx);
            ckpt.update_pos((int64_t) n_past,
                            llama_memory_seq_pos_min(mem, 0),
                            llama_memory_seq_pos_max(mem, 0));
            ckpt.update_tgt(g_ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (g_ctx_dft) {
                ckpt.update_dft(g_ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
        }

        // Build [id_last, draft0..draftN-1] and evaluate on the target.
        common_batch_clear(g_spec_batch);
        std::vector<int> idxs;
        idxs.reserve(draft.size() + 1);
        common_batch_add(g_spec_batch, id_last, n_past, { 0 }, true);
        idxs.push_back(0);
        for (size_t k = 0; k < draft.size(); ++k) {
            common_batch_add(g_spec_batch, draft[k], n_past + 1 + (llama_pos) k, { 0 }, true);
            idxs.push_back((int) (k + 1));
        }
        if (llama_decode(g_ctx, g_spec_batch) != 0) {
            log_to_file("spec_decode_step: target decode failed", GGML_LOG_LEVEL_ERROR);
            return false;
        }

        g_spec_stat_eval += (long) g_spec_batch.n_tokens;  // positions actually decoded

        // Feed the decoded batch to the MTP speculator (extracts the nextn hidden states).
        common_speculative_process(g_spec, g_spec_batch);

        // Verify against the target sampler; accept the matching prefix (+1 bonus token).
        common_sampler_ptr smpl_save;
        if (use_ckpt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }
        out_ids = common_sampler_sample_and_accept_n(smpl.get(), g_ctx, idxs, draft);
        if (out_ids.empty()) {
            log_to_file("spec_decode_step: sample_and_accept_n returned empty", GGML_LOG_LEVEL_WARN);
            return false;
        }
        const uint32_t n_rollback = (uint32_t) draft.size() + 1u - (uint32_t) out_ids.size();

        if (n_rollback > 0 && use_ckpt) {
            // FULL context: restore to the checkpoint (before this draft) and re-evaluate
            // only the accepted tokens next round; they are the target's own samples, so
            // they re-accept -> guaranteed forward progress within max_retries.
            ckpt.load_tgt(g_ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            common_context_seq_rm(g_ctx, 0, ckpt.pos_max + 1, -1);
            if (g_ctx_dft) {
                ckpt.load_dft(g_ctx_dft, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                common_context_seq_rm(g_ctx_dft, 0, ckpt.pos_max + 1, -1);
            }
            smpl = std::move(smpl_save);            // restore the pre-verify sampler state
            draft = (attempt >= max_retries) ? std::vector<llama_token>{} : out_ids;
            reuse_draft = true;
            continue;
        }

        // Accept and commit.
        common_speculative_accept(g_spec, 0, (uint16_t) (out_ids.size() - 1));

        if (!use_ckpt) {
            // PART context: drop the rejected draft tail directly.
            const llama_pos keep = n_past + (llama_pos) out_ids.size();
            common_context_seq_rm(g_ctx, 0, keep, -1);
            if (g_ctx_dft) {
                common_context_seq_rm(g_ctx_dft, 0, keep, -1);
            }
        }

        g_spec_stat_steps++;
        g_spec_stat_out += (long) out_ids.size();
        n_past += (llama_pos) out_ids.size();
        id_last = out_ids.back();
        return true;
    }
}

static bool reset_generation_context_locked(const char * log_prefix, std::string & error_message) {
    if (!g_model || !g_ctx) {
        error_message = "not initialized";
        return false;
    }

    // Each generation rebuilds the full prompt and is independent, so the persistent context
    // is simply reset to a clean state between calls. Clearing the memory resets the KV cache
    // and any recurrent/hybrid (SSM) state, which is sufficient and far cheaper than the
    // previous per-generation context teardown/recreate workaround.
    llama_synchronize(g_ctx);

    llama_memory_t memory = llama_get_memory(g_ctx);
    if (!memory) {
        error_message = "generation memory unavailable";
        log_to_file(std::string(log_prefix) + ": " + error_message, GGML_LOG_LEVEL_ERROR);
        return false;
    }
    llama_memory_clear(memory, true);
    g_kv_cached_tokens.clear();   // KV is now empty -> prefix cache invalid
    llama_perf_context_reset(g_ctx);

    std::ostringstream ss;
    ss << log_prefix << ": generation context reset"
       << " n_ctx=" << g_n_ctx
       << " n_batch=" << g_n_batch
       << " n_threads=" << g_n_threads;
    log_to_file(ss.str());
    return true;
}

static void log_to_file(const std::string& msg, ggml_log_level level) {
    if (!should_log(level)) return;
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_path.empty()) return;
    if (!g_log_ofs.is_open()) {
        g_log_ofs.open(g_log_path, std::ios::app | std::ios::binary);
    }
    if (!g_log_ofs) return;
    g_log_ofs << current_time_str() << " [JNI] " << msg << std::endl;
    g_log_ofs.flush();
}

static void trim_native_allocator(const char * reason) {
#if defined(__GLIBC__)
    const int trimmed = malloc_trim(0);
    std::ostringstream ss;
    ss << reason << ": malloc_trim result=" << trimmed;
    log_to_file(ss.str());
#elif defined(__ANDROID__)
    using mallopt_fn = int (*)(int, int);
    mallopt_fn mallopt_ptr = reinterpret_cast<mallopt_fn>(dlsym(RTLD_DEFAULT, "mallopt"));
    if (mallopt_ptr == nullptr) {
        if (reason != nullptr) {
            log_to_file(std::string(reason) + ": mallopt unavailable on this platform");
        }
        return;
    }

    int purge_all_result = 0;
    int purge_result = 0;
#if defined(M_PURGE_ALL)
    purge_all_result = mallopt_ptr(M_PURGE_ALL, 0);
#endif
#if defined(M_PURGE)
    purge_result = mallopt_ptr(M_PURGE, 0);
#endif
    std::ostringstream ss;
    ss << reason
       << ": mallopt purge_all=" << purge_all_result
       << " purge=" << purge_result;
    log_to_file(ss.str());
#else
    if (reason != nullptr) {
        log_to_file(std::string(reason) + ": malloc_trim unavailable on this platform");
    }
#endif
}

// The per-second native memory sampler was removed: memory is snapshotted at every
// generation boundary by the Java diagnostics (ModelManager generation-start/end), and the
// authoritative cause of any real process death is captured by ApplicationExitInfo plus the
// fatal-signal handler. An always-on sampling thread per generation was pure overhead.

static void ensure_backend_initialized_locked(const char * log_prefix) {
    // Re-initialize if config changed since last init (e.g. NPU toggled)
    if (g_backend_initialized && g_backend_initialized_version == g_backend_config_version) {
        std::ostringstream ss;
        ss << log_prefix << ": backend already initialized (version=" << g_backend_config_version
           << "), reusing";
        log_to_file(ss.str());
        return;
    }

    if (g_backend_initialized) {
        std::ostringstream ss;
        ss << log_prefix << ": backend config changed (version "
           << g_backend_initialized_version << " -> " << g_backend_config_version
           << "), re-initializing";
        log_to_file(ss.str());
        release_backend_locked(log_prefix);
    }

#if defined(GGML_USE_HEXAGON)
    const bool npu_active = g_npu_enabled && (g_backend_type == 2 || g_backend_type == 3);
    // Set ADSP_LIBRARY_PATH UNCONDITIONALLY (not only for NPU) and BEFORE llama_backend_init():
    // the Hexagon backend is auto-registered by the ggml registry on the first backend init
    // regardless of the selected backend, and that registration eagerly opens a DSP session
    // which loads libggml-htp-vNN.so via the FastRPC loader. If the path is unset at that moment
    // (e.g. the app's first init is CPU mode at startup), the skel is not found, the session
    // fails (error 0x80000406), and ggml_backend_hexagon_reg() caches that failure for the whole
    // process — so a later switch to NPU never finds a Hexagon device and silently falls back to
    // CPU. Setting it here lets the very first session attempt succeed and keeps NPU available.
    if (!g_native_lib_dir.empty()) {
        if (setenv("ADSP_LIBRARY_PATH", g_native_lib_dir.c_str(), 1) == 0) {
            log_to_file(std::string(log_prefix) + ": set ADSP_LIBRARY_PATH=" + g_native_lib_dir);
        } else {
            log_to_file(std::string(log_prefix) + ": setenv(ADSP_LIBRARY_PATH) failed",
                        GGML_LOG_LEVEL_WARN);
        }
    }
    if (npu_active && !g_native_lib_dir.empty()) {
        // 診断: HTP skel が nativeLibraryDir に「ファイルとして」存在するか確認する。
        // 1 つも無ければ extractNativeLibs/useLegacyPackaging が効いておらず NPU は必ず失敗する。
        int skel_found = 0;
        for (const char * v : {"v68", "v69", "v73", "v75", "v79", "v81"}) {
            const std::string p = g_native_lib_dir + "/libggml-htp-" + v + ".so";
            if (access(p.c_str(), R_OK) == 0) {
                ++skel_found;
                log_to_file(std::string(log_prefix) + ": HTP skel present: " + p);
            }
        }
        if (skel_found == 0) {
            log_to_file(std::string(log_prefix) + ": no HTP skel files found under " +
                        g_native_lib_dir + " — NPU cannot initialize (check extractNativeLibs)",
                        GGML_LOG_LEVEL_ERROR);
        }
    }
#endif

    llama_backend_init();
    g_backend_initialized = true;

#if defined(GGML_USE_HEXAGON)
    if (g_npu_enabled && (g_backend_type == 2 || g_backend_type == 3)) {
        ggml_backend_reg_t hex_reg = ggml_backend_hexagon_reg();
        if (hex_reg) {
            ggml_backend_register(hex_reg);
            log_to_file(std::string(log_prefix) + ": ggml: Hexagon/HTP backend registered");
        } else {
            log_to_file(std::string(log_prefix) + ": ggml: Hexagon backend unavailable on this device",
                        GGML_LOG_LEVEL_WARN);
        }
    }
#endif

    // OpenCL (Adreno GPU) backend is shipped as a DL plugin (libggml-opencl.so) so the
    // app does not hard-depend on libOpenCL.so. Load + register it on demand when GPU is
    // selected. Registered once per process; a missing OpenCL driver fails gracefully.
    if (g_backend_type == 1 || g_backend_type == 3) {
        static bool g_opencl_registered = false;
        if (!g_opencl_registered && !g_native_lib_dir.empty()) {
            const std::string ocl_path = g_native_lib_dir + "/libggml-opencl.so";
            void * h = dlopen(ocl_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if (!h) {
                const char * e = dlerror();
                log_to_file(std::string(log_prefix) + ": OpenCL plugin load failed: " +
                            (e ? e : "unknown"), GGML_LOG_LEVEL_WARN);
            } else {
                using ocl_reg_fn = ggml_backend_reg_t (*)(void);
                ocl_reg_fn init_fn = reinterpret_cast<ocl_reg_fn>(dlsym(h, "ggml_backend_init"));
                if (init_fn) {
                    ggml_backend_reg_t reg = init_fn();
                    if (reg) {
                        ggml_backend_register(reg);
                        g_opencl_registered = true;
                        log_to_file(std::string(log_prefix) + ": ggml: OpenCL/GPU backend registered");
                    } else {
                        log_to_file(std::string(log_prefix) + ": OpenCL plugin returned null reg",
                                    GGML_LOG_LEVEL_WARN);
                    }
                } else {
                    log_to_file(std::string(log_prefix) + ": OpenCL plugin missing ggml_backend_init",
                                GGML_LOG_LEVEL_WARN);
                }
            }
        }
    }

    g_backend_initialized_version = g_backend_config_version;

    std::ostringstream ss;
    ss << log_prefix << ": backend init complete (version=" << g_backend_config_version << "), "
       << summarize_registered_backends();
    const size_t backend_count = ggml_backend_reg_count();
    log_to_file(ss.str(), backend_count == 0 ? GGML_LOG_LEVEL_ERROR : GGML_LOG_LEVEL_INFO);
}

// ---------------- llama.cpp ログコールバック ----------------
// 0.17.1 は llama_log_level ではなく ggml_log_level を使う
static void llama_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    try {
        const ggml_log_level effective = normalize_log_level(level);
        if (!should_log(effective)) return;
        std::string msg = text ? text : "";
        // Skip empty messages and continuation messages
        if (msg.empty() || msg == "\n") {
            return;
        }
        // Only log INFO, WARN, ERROR levels
        if (effective == GGML_LOG_LEVEL_ERROR) {
            LOGE("[llama.cpp] %s", msg.c_str());
        } else if (effective == GGML_LOG_LEVEL_WARN) {
            LOGI("[llama.cpp WARN] %s", msg.c_str());
        } else if (effective == GGML_LOG_LEVEL_DEBUG) {
            LOGI("[llama.cpp DEBUG] %s", msg.c_str());
        } else {
            LOGI("[llama.cpp] %s", msg.c_str());
        }
        log_to_file(std::string("llama.cpp: ") + msg, effective);
    } catch (const std::exception & e) {
        LOGE("llama_log_callback exception: %s", e.what());
    } catch (...) {
        LOGE("llama_log_callback unknown exception");
    }
}

// ---------------- 既存ユーティリティ ----------------
static std::string jstring_to_std(JNIEnv *env, jstring jstr) {
    if (!jstr) return "";
    const char *chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars ? chars : "");
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

static std::string to_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static std::string path_filename(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static bool is_regular_file_path(const std::string & path) {
    struct stat st = {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

static std::string strip_gguf_suffix(const std::string & filename) {
    std::string lower = to_lower_ascii(path_filename(filename));
    if (lower.size() > 5 && lower.compare(lower.size() - 5, 5, ".gguf") == 0) {
        lower.resize(lower.size() - 5);
    }
    return lower;
}

static bool contains_any_substring(const std::string & haystack, const std::vector<std::string> & needles) {
    for (const auto & needle : needles) {
        if (!needle.empty() && haystack.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static std::vector<std::string> collect_multimodal_projector_candidates(
        const std::string & model_path,
        const std::string & requested_mmproj_path) {
    (void) model_path;
    std::vector<std::string> result;
    // Only try the projector path that Java has already resolved.
    // Native directory scanning made text-only model switches pick unrelated mmproj files
    // that happened to live next to the model, which could crash during mtmd init.
    if (!requested_mmproj_path.empty() && is_regular_file_path(requested_mmproj_path)) {
        result.push_back(requested_mmproj_path);
    }
    return result;
}

static bool is_likely_multimodal_model_path(const std::string & model_path) {
    const std::string lower_name = strip_gguf_suffix(model_path);
    return contains_any_substring(lower_name, {
            "gemma-4", "gemma4", "gemma-3n", "gemma3n", "llava", "vision",
            "audio", "qwen2-vl", "qwen25vl", "qwen2vl", "qwen2.5-vl",
            "qwen2-audio", "qwen2audio", "qwen2.5-omni", "qwen25omni",
            "glm-4v", "glm4v", "minicpm-v", "minicpmv", "voxtral", "ultravox"
    });
}

static std::string initialize_optional_multimodal_support_locked(
        const std::string & model_path,
        const std::string & requested_mmproj_path,
        bool enable_audio,
        const char * log_prefix) {
    if (g_mtmd) {
        mtmd_free(g_mtmd);
        g_mtmd = nullptr;
    }
    g_supports_vision = false;
    g_supports_audio = false;

    const bool likely_multimodal_model = is_likely_multimodal_model_path(model_path);
    const auto candidates = collect_multimodal_projector_candidates(model_path, requested_mmproj_path);

    if (candidates.empty()) {
        if (!requested_mmproj_path.empty()) {
            std::ostringstream ss;
            ss << log_prefix << ": requested mmproj is unavailable or not a regular file: "
               << requested_mmproj_path << "; loading text-only";
            log_to_file(ss.str(), GGML_LOG_LEVEL_WARN);
        } else if (likely_multimodal_model) {
            std::ostringstream ss;
            ss << log_prefix << ": likely multimodal model detected but no explicit mmproj was requested; "
               << "skipping native projector auto-detection for model " << model_path
               << "; multimodal support disabled";
            log_to_file(ss.str(), GGML_LOG_LEVEL_WARN);
        }
        return "";
    }

    for (const auto & candidate_path : candidates) {
        std::ostringstream start_ss;
        start_ss << log_prefix << ": trying multimodal projector " << candidate_path;
        log_to_file(start_ss.str());

        mtmd_context_params mparams_mtmd = mtmd_context_params_default();
        mparams_mtmd.use_gpu = g_n_gpu_layers != 0;
        mparams_mtmd.print_timings = false;
        mparams_mtmd.n_threads = g_n_threads;
        mparams_mtmd.warmup = true;
        mparams_mtmd.skip_audio = !enable_audio;

        mtmd_context * mtmd = mtmd_init_from_file(candidate_path.c_str(), g_model, mparams_mtmd);
        if (!mtmd) {
            std::ostringstream fail_ss;
            fail_ss << log_prefix << ": failed to initialize multimodal projector " << candidate_path
                    << "; trying next candidate if available";
            log_to_file(fail_ss.str(), GGML_LOG_LEVEL_WARN);
            continue;
        }

        const bool supports_vision = mtmd_support_vision(mtmd);
        const bool supports_audio = mtmd_support_audio(mtmd);
        if (!supports_vision && !supports_audio) {
            std::ostringstream fail_ss;
            fail_ss << log_prefix << ": projector initialized but exposed no supported modalities: "
                    << candidate_path;
            log_to_file(fail_ss.str(), GGML_LOG_LEVEL_WARN);
            mtmd_free(mtmd);
            continue;
        }

        g_mtmd = mtmd;
        g_supports_vision = supports_vision;
        g_supports_audio = supports_audio;

        std::ostringstream ok_ss;
        ok_ss << log_prefix << ": multimodal projector loaded path=" << candidate_path
              << " vision=" << g_supports_vision
              << " audio=" << g_supports_audio;
        log_to_file(ok_ss.str());
        return candidate_path;
    }

    if (likely_multimodal_model || !requested_mmproj_path.empty()) {
        std::ostringstream ss;
        ss << log_prefix << ": no usable multimodal projector could be initialized for model "
           << model_path << "; multimodal support disabled";
        log_to_file(ss.str(), GGML_LOG_LEVEL_WARN);
    }
    return "";
}

static void throw_java_exception(JNIEnv *env, const char *msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass) env->ThrowNew(exClass, msg);
}

// Helper function to process escape sequences in a string
// Converts user-friendly escape sequences like "\n" (backslash+n) to actual characters (newline)
static std::string process_escape_sequences(const std::string& input) {
    std::string result;
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '\\' && i + 1 < input.size()) {
            switch (input[i + 1]) {
                case 'n': result += '\n'; i++; break;
                case 't': result += '\t'; i++; break;
                case 'r': result += '\r'; i++; break;
                case '\\': result += '\\'; i++; break;
                case '"': result += '"'; i++; break;
                default: result += input[i]; break;
            }
        } else {
            result += input[i];
        }
    }
    return result;
}

static std::vector<std::string> parse_dry_sequence_breakers() {
    std::vector<std::string> breaker_strings;
    std::string temp = g_dry_sequence_breakers;
    size_t pos = 0;
    while ((pos = temp.find(',')) != std::string::npos) {
        std::string token = temp.substr(0, pos);
        if (!token.empty()) {
            breaker_strings.push_back(process_escape_sequences(token));
        }
        temp.erase(0, pos + 1);
    }
    if (!temp.empty()) {
        breaker_strings.push_back(process_escape_sequences(temp));
    }
    return breaker_strings;
}

static bool find_stop_sequence_in_list(
        const std::vector<std::string> & stop_sequences,
        const std::string & text,
        size_t * stop_pos,
        std::string * stop_seq) {
    size_t earliest = std::string::npos;
    std::string found;
    for (const auto & seq : stop_sequences) {
        if (seq.empty()) {
            continue;
        }
        size_t pos = text.find(seq);
        if (pos != std::string::npos && (earliest == std::string::npos || pos < earliest)) {
            earliest = pos;
            found = seq;
        }
    }
    if (earliest != std::string::npos) {
        if (stop_pos) *stop_pos = earliest;
        if (stop_seq) *stop_seq = found;
        return true;
    }
    return false;
}

static common_params_sampling build_chat_sampling_params_locked(const common_chat_params & chat_params) {
    common_params_sampling params;
    params.seed               = LLAMA_DEFAULT_SEED;
    params.top_k              = g_top_k;
    params.top_p              = g_top_p;
    params.min_p              = g_min_p;
    params.xtc_probability    = g_xtc_probability;
    params.xtc_threshold      = g_xtc_threshold;
    params.typ_p              = g_typical_p;
    params.temp               = g_temp;
    params.dynatemp_range     = g_dynatemp_range;
    params.dynatemp_exponent  = g_dynatemp_exponent;
    params.penalty_last_n     = g_penalty_last_n > 0 ? g_penalty_last_n : g_n_ctx;
    params.penalty_repeat     = g_penalty_repeat;
    params.penalty_freq       = g_penalty_freq;
    params.penalty_present    = g_penalty_present;
    params.dry_multiplier     = g_dry_multiplier;
    params.dry_base           = g_dry_base;
    params.dry_allowed_length = g_dry_allowed_length;
    params.dry_penalty_last_n = g_dry_penalty_last_n > 0 ? g_dry_penalty_last_n : g_n_ctx;
    params.mirostat           = g_mirostat;
    params.mirostat_tau       = g_mirostat_tau;
    params.mirostat_eta       = g_mirostat_eta;
    params.top_n_sigma        = g_top_n_sigma;
    params.dry_sequence_breakers = parse_dry_sequence_breakers();
    params.n_prev             = std::max(32, params.penalty_last_n);
    params.grammar_lazy       = chat_params.grammar_lazy;
    params.generation_prompt  = chat_params.generation_prompt;
    params.backend_sampling   = false;

    if (!chat_params.grammar.empty()) {
        params.grammar = { COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat_params.grammar };
    }
    for (const auto & trigger : chat_params.grammar_triggers) {
        params.grammar_triggers.push_back(trigger);
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    for (const auto & token_text : chat_params.preserved_tokens) {
        auto ids = common_tokenize(vocab, token_text, false, true);
        if (ids.size() == 1) {
            params.preserved_tokens.insert(ids[0]);
        }
    }
    return params;
}

static std::vector<std::string> build_chat_stop_sequences(const common_chat_params & chat_params) {
    std::vector<std::string> stop_sequences = DEFAULT_STOP_SEQUENCES;
    for (const auto & stop : chat_params.additional_stops) {
        if (stop.empty()) {
            continue;
        }
        if (std::find(stop_sequences.begin(), stop_sequences.end(), stop) == stop_sequences.end()) {
            stop_sequences.push_back(stop);
        }
    }
    return stop_sequences;
}

// ---------------- download() 用 ----------------
static size_t write_data(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::ofstream* ofs = reinterpret_cast<std::ofstream*>(userdata);
    if (!ofs) {
        return 0;
    }
    const size_t bytes = size * nmemb;
    ofs->write(reinterpret_cast<const char*>(ptr), bytes);
    if (!(*ofs)) {
        log_to_file("download: file write failed", GGML_LOG_LEVEL_ERROR);
        return 0;
    }
    return bytes;
}

struct ProgressData {
    jobject thiz_global;
    jmethodID onProgressMethod;
    int last_percent;
};

static int xferinfo(void* p, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
    ProgressData* pd = reinterpret_cast<ProgressData*>(p);
    if (!pd) return 0;
    if (dltotal <= 0) return 0;

    int percent = (int)((dlnow * 100) / dltotal);
    if (percent == pd->last_percent) return 0;
    pd->last_percent = percent;

    if (!g_jvm) return 0;

    JNIEnv* env = nullptr;
    bool attached = false;
    if (g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
            return 0;
        }
        attached = true;
    }

    if (env && pd->thiz_global && pd->onProgressMethod) {
        env->CallVoidMethod(pd->thiz_global, pd->onProgressMethod, (jint)percent);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }

    {
        std::ostringstream ss;
        ss << "Download progress: " << percent << "%";
        log_to_file(ss.str());
    }

    if (attached) g_jvm->DetachCurrentThread();
    return 0;
}

static bool is_existing_nonempty_file(const std::string & path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const std::streampos size = ifs.tellg();
    return size > 0;
}

static bool is_ascii_digits(const std::string & value) {
    if (value.empty()) {
        return false;
    }
    for (unsigned char c : value) {
        if (!std::isdigit(c)) {
            return false;
        }
    }
    return true;
}

static bool parse_split_file_info(
        const std::string & candidate,
        std::string & prefix,
        int & split_no,
        int & split_count) {
    static const std::string suffix = ".gguf";
    if (candidate.size() <= suffix.size() ||
        candidate.compare(candidate.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }

    const size_t suffix_pos = candidate.size() - suffix.size();
    const size_t of_pos = candidate.rfind("-of-", suffix_pos);
    if (of_pos == std::string::npos) {
        return false;
    }

    const size_t no_pos = candidate.rfind('-', of_pos == 0 ? 0 : of_pos - 1);
    if (no_pos == std::string::npos) {
        return false;
    }

    const std::string split_no_str = candidate.substr(no_pos + 1, of_pos - (no_pos + 1));
    const std::string split_count_str = candidate.substr(of_pos + 4, suffix_pos - (of_pos + 4));
    if (split_no_str.size() != 5 || split_count_str.size() != 5 ||
        !is_ascii_digits(split_no_str) || !is_ascii_digits(split_count_str)) {
        return false;
    }

    split_no = std::stoi(split_no_str);
    split_count = std::stoi(split_count_str);
    if (split_no <= 0 || split_count <= 1 || split_no > split_count) {
        return false;
    }

    prefix = candidate.substr(0, no_pos);
    return !prefix.empty();
}

static std::string split_url_suffix(const std::string & url, std::string & url_without_suffix) {
    const size_t query_pos = url.find('?');
    const size_t fragment_pos = url.find('#');

    size_t suffix_pos = std::string::npos;
    if (query_pos != std::string::npos && fragment_pos != std::string::npos) {
        suffix_pos = std::min(query_pos, fragment_pos);
    } else if (query_pos != std::string::npos) {
        suffix_pos = query_pos;
    } else {
        suffix_pos = fragment_pos;
    }

    if (suffix_pos == std::string::npos) {
        url_without_suffix = url;
        return "";
    }

    url_without_suffix = url.substr(0, suffix_pos);
    return url.substr(suffix_pos);
}

static void cleanup_downloaded_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            continue;
        }
        if (unlink(path.c_str()) == 0) {
            log_to_file(std::string("download: removed partial file ") + path, GGML_LOG_LEVEL_WARN);
        }
    }
}

static bool download_file_with_curl(
        const std::string & url,
        const std::string & path,
        ProgressData * progress_data,
        std::string & error_message) {
    CURL * curl = curl_easy_init();
    if (!curl) {
        error_message = "curl init failed";
        log_to_file("download: curl init failed", GGML_LOG_LEVEL_ERROR);
        return false;
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        curl_easy_cleanup(curl);
        error_message = "file open failed";
        log_to_file(std::string("download: file open failed path=") + path, GGML_LOG_LEVEL_ERROR);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ofs);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    std::string ca_bundle_path;
    {
        std::lock_guard<std::mutex> lock(g_download_ca_bundle_mutex);
        ca_bundle_path = g_download_ca_bundle_path;
    }
    if (!ca_bundle_path.empty()) {
        curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle_path.c_str());
        log_to_file(std::string("download: using CA bundle path=") + ca_bundle_path);
    } else {
        log_to_file("download: no CA bundle configured; using libcurl defaults", GGML_LOG_LEVEL_WARN);
    }

    if (progress_data) {
        progress_data->last_percent = -1;
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xferinfo);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, progress_data);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    } else {
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    }

    curl_easy_setopt(curl, CURLOPT_USERAGENT,
        "Mozilla/5.0 (Linux; Android 14; Mobile) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Mobile Safari/537.36");

    const CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    ofs.flush();
    ofs.close();
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::ostringstream ss;
        ss << "curl download failed";
        if (http_code > 0) {
            ss << " (HTTP " << http_code << ")";
        }
        ss << ": " << curl_easy_strerror(res);
        error_message = ss.str();
        log_to_file(std::string("download: ") + ss.str(), GGML_LOG_LEVEL_ERROR);
        unlink(path.c_str());
        return false;
    }

    if (!is_existing_nonempty_file(path)) {
        error_message = "downloaded file is empty";
        log_to_file(std::string("download: empty file after download path=") + path, GGML_LOG_LEVEL_ERROR);
        unlink(path.c_str());
        return false;
    }

    return true;
}

static bool read_split_count_from_gguf(
        const std::string & path,
        int & n_split,
        std::string & error_message) {
    n_split = 0;

    struct gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ nullptr,
    };

    gguf_context * ctx_gguf = gguf_init_from_file(path.c_str(), gguf_params);
    if (!ctx_gguf) {
        error_message = std::string("failed to load GGUF metadata from ") + path;
        log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
        return false;
    }

    const int64_t key_n_split = gguf_find_key(ctx_gguf, "split.count");
    if (key_n_split >= 0) {
        n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
    }

    gguf_free(ctx_gguf);
    return true;
}

// ---- mmproj <-> model compatibility pre-check (request #6) --------------------
// Reading GGUF *metadata only* (no_alloc) lets us catch the common "wrong mmproj"
// mistakes BEFORE mtmd_init_from_file. mtmd catches std::exceptions (e.g. the
// explicit n_embd mismatch throw), but a structurally wrong projector trips a
// GGML_ASSERT during clip construction, which calls abort() and cannot be caught
// in Java — so we must reject it up-front instead.

static bool gguf_read_uint_kv_as_i64(
        const gguf_context * ctx, const char * key, int64_t & out_value) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return false;
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT32: out_value = (int64_t) gguf_get_val_u32(ctx, kid); return true;
        case GGUF_TYPE_INT32:  out_value = (int64_t) gguf_get_val_i32(ctx, kid); return true;
        case GGUF_TYPE_UINT64: out_value = (int64_t) gguf_get_val_u64(ctx, kid); return true;
        case GGUF_TYPE_INT64:  out_value = (int64_t) gguf_get_val_i64(ctx, kid); return true;
        case GGUF_TYPE_UINT16: out_value = (int64_t) gguf_get_val_u16(ctx, kid); return true;
        case GGUF_TYPE_INT16:  out_value = (int64_t) gguf_get_val_i16(ctx, kid); return true;
        default:               return false;
    }
}

static bool gguf_kv_is_true(const gguf_context * ctx, const char * key) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_BOOL) {
        return false;
    }
    return gguf_get_val_bool(ctx, kid);
}

// Returns "ok" | "incompatible:..." | "unknown". Confident answers only; anything
// it cannot determine is reported as "unknown" so valid combinations are never blocked.
static std::string validate_mmproj_for_model(
        const std::string & model_path,
        const std::string & mmproj_path) {
    if (mmproj_path.empty()) {
        return "unknown";
    }

    // Load mmproj metadata + tensor shapes (no_alloc => cheap, no tensor data, no asserts).
    struct ggml_context * mmproj_ggml = nullptr;
    struct gguf_init_params mmproj_params = { /*.no_alloc =*/ true, /*.ctx =*/ &mmproj_ggml };
    gguf_context * mmproj_gguf = gguf_init_from_file(mmproj_path.c_str(), mmproj_params);
    if (!mmproj_gguf) {
        return "unknown";
    }

    std::string result = "unknown";
    int64_t model_n_embd = -1;
    bool model_known = false;

    // A genuine mmproj/CLIP file always advertises a vision and/or audio encoder.
    // Otherwise the user picked a normal model file as the projector — a clear error.
    const bool is_mmproj =
            gguf_kv_is_true(mmproj_gguf, "clip.has_vision_encoder") ||
            gguf_kv_is_true(mmproj_gguf, "clip.has_audio_encoder");

    if (!is_mmproj) {
        result = "incompatible:not_mmproj";
    } else {
        // Collect raw dims of the multimodal projector ("mm.*") tensors. The projector
        // output dim (which must equal the text model's embedding length) is always one
        // of these raw ne values, regardless of projector type — so membership is a safe,
        // type-agnostic compatibility test with no false positives for "mm.*" projectors.
        std::vector<int64_t> mm_dims;
        if (mmproj_ggml) {
            const int64_t n_tensors = gguf_get_n_tensors(mmproj_gguf);
            for (int64_t i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(mmproj_gguf, i);
                if (!name || std::strncmp(name, "mm.", 3) != 0) {
                    continue;
                }
                const ggml_tensor * t = ggml_get_tensor(mmproj_ggml, name);
                if (t) {
                    mm_dims.push_back(t->ne[0]);
                    mm_dims.push_back(t->ne[1]);
                }
            }
        }

        // Read the text model's embedding length from its GGUF metadata.
        struct gguf_init_params model_params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
        gguf_context * model_gguf = gguf_init_from_file(model_path.c_str(), model_params);
        if (model_gguf) {
            const int64_t arch_kid = gguf_find_key(model_gguf, "general.architecture");
            if (arch_kid >= 0 && gguf_get_kv_type(model_gguf, arch_kid) == GGUF_TYPE_STRING) {
                const std::string arch = gguf_get_val_str(model_gguf, arch_kid);
                const std::string embd_key = arch + ".embedding_length";
                model_known = gguf_read_uint_kv_as_i64(model_gguf, embd_key.c_str(), model_n_embd)
                              && model_n_embd > 0;
            }
            gguf_free(model_gguf);
        }

        if (!model_known || mm_dims.empty()) {
            // Could not read the model embd, or projector stores its output under a
            // non-"mm." prefix (e.g. minicpmv resampler.*). Don't guess.
            result = "unknown";
        } else {
            bool matched = false;
            for (int64_t d : mm_dims) {
                if (d == model_n_embd) {
                    matched = true;
                    break;
                }
            }
            if (matched) {
                result = "ok";
            } else {
                std::ostringstream ss;
                ss << "incompatible:embd:model=" << model_n_embd;
                result = ss.str();
            }
        }
    }

    gguf_free(mmproj_gguf);
    if (mmproj_ggml) {
        ggml_free(mmproj_ggml);
    }

    std::ostringstream log;
    log << "validateMmproj: model=" << path_filename(model_path)
        << " mmproj=" << path_filename(mmproj_path)
        << " result=" << result;
    if (model_known) {
        log << " n_embd=" << model_n_embd;
    }
    log_to_file(log.str());

    return result;
}

static bool find_missing_split_shard_path(
        const std::string & model_path,
        std::string & missing_path) {
    std::string split_prefix;
    int split_no = 0;
    int split_count = 0;
    if (!parse_split_file_info(model_path, split_prefix, split_no, split_count)) {
        missing_path.clear();
        return false;
    }

    for (int idx = 0; idx < split_count; ++idx) {
        std::vector<char> split_path_buf(split_prefix.size() + 32, '\0');
        if (!llama_split_path(split_path_buf.data(), split_path_buf.size(), split_prefix.c_str(), idx, split_count)) {
            missing_path = model_path;
            return true;
        }
        const std::string shard_path(split_path_buf.data());
        if (!is_existing_nonempty_file(shard_path)) {
            missing_path = shard_path;
            return true;
        }
    }

    missing_path.clear();
    return false;
}

static bool download_model_with_splits(
        const std::string & model_url,
        const std::string & model_path,
        ProgressData * progress_data,
        std::string & error_message) {
    std::vector<std::string> newly_downloaded_paths;
    bool allow_progress = progress_data != nullptr;

    if (!is_existing_nonempty_file(model_path)) {
        log_to_file(std::string("download: fetching primary shard url=") + model_url + " path=" + model_path);
        if (!download_file_with_curl(model_url, model_path, allow_progress ? progress_data : nullptr, error_message)) {
            cleanup_downloaded_files(newly_downloaded_paths);
            return false;
        }
        newly_downloaded_paths.push_back(model_path);
        allow_progress = false;
    } else {
        log_to_file(std::string("download: primary shard already present, probing metadata path=") + model_path);
    }

    int n_split = 0;
    if (!read_split_count_from_gguf(model_path, n_split, error_message)) {
        cleanup_downloaded_files(newly_downloaded_paths);
        return false;
    }

    if (n_split <= 1) {
        return true;
    }

    std::string path_prefix;
    int split_no = 0;
    int split_count_from_name = 0;
    if (!parse_split_file_info(model_path, path_prefix, split_no, split_count_from_name)) {
        error_message = std::string("unexpected split model file name: ") + model_path;
        log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
        cleanup_downloaded_files(newly_downloaded_paths);
        return false;
    }
    if (split_count_from_name != n_split) {
        error_message = std::string("split count mismatch between file name and GGUF metadata: ") + model_path;
        log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
        cleanup_downloaded_files(newly_downloaded_paths);
        return false;
    }

    std::string url_without_suffix;
    const std::string url_suffix = split_url_suffix(model_url, url_without_suffix);
    std::string url_prefix;
    int url_split_no = 0;
    int url_split_count = 0;
    if (!parse_split_file_info(url_without_suffix, url_prefix, url_split_no, url_split_count)) {
        error_message = std::string("unexpected split model url: ") + model_url;
        log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
        cleanup_downloaded_files(newly_downloaded_paths);
        return false;
    }
    if (url_split_no != split_no || url_split_count != n_split) {
        error_message = std::string("split model url does not match local shard naming: ") + model_url;
        log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
        cleanup_downloaded_files(newly_downloaded_paths);
        return false;
    }

    {
        std::ostringstream ss;
        ss << "download: split GGUF detected split_no=" << split_no << " split_count=" << n_split;
        log_to_file(ss.str());
    }

    for (int idx = 0; idx < n_split; ++idx) {
        std::vector<char> split_path_buf(path_prefix.size() + 32, '\0');
        std::vector<char> split_url_buf(url_prefix.size() + 32, '\0');
        if (!llama_split_path(split_path_buf.data(), split_path_buf.size(), path_prefix.c_str(), idx, n_split) ||
            !llama_split_path(split_url_buf.data(), split_url_buf.size(), url_prefix.c_str(), idx, n_split)) {
            error_message = "failed to build split file path";
            log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
            cleanup_downloaded_files(newly_downloaded_paths);
            return false;
        }

        const std::string split_path(split_path_buf.data());
        if (is_existing_nonempty_file(split_path)) {
            continue;
        }

        const std::string split_url = std::string(split_url_buf.data()) + url_suffix;
        log_to_file(std::string("download: fetching split shard url=") + split_url + " path=" + split_path);
        if (!download_file_with_curl(split_url, split_path, allow_progress ? progress_data : nullptr, error_message)) {
            cleanup_downloaded_files(newly_downloaded_paths);
            return false;
        }
        newly_downloaded_paths.push_back(split_path);
        allow_progress = false;
    }

    std::string missing_split_path;
    if (find_missing_split_shard_path(model_path, missing_split_path)) {
        error_message = std::string("missing split shard after download: ") + missing_split_path;
        log_to_file(std::string("download: ") + error_message, GGML_LOG_LEVEL_ERROR);
        cleanup_downloaded_files(newly_downloaded_paths);
        return false;
    }

    return true;
}

// ---------------- 解放 ----------------
static void llama_jni_free() {
    std::lock_guard<std::mutex> lock(g_mutex);

    log_to_file("llama_jni_free: freeing resources (explicit)");

    release_multimodal_locked("llama_jni_free");
    release_context_locked("llama_jni_free");
    release_model_locked("llama_jni_free");
    release_backend_locked("llama_jni_free");
    g_current_mmproj_path.clear();
    g_supports_vision = false;
    g_supports_audio = false;
    g_current_audio_requested = false;

    trim_native_allocator("llama_jni_free");
}

// ---------------- JNI: setLogPath ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setLogPath(
        JNIEnv *env, jobject, jstring jLogPath) {

    std::string path = jstring_to_std(env, jLogPath);
    ensure_fatal_signal_handlers_installed();
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (path == g_log_path && g_log_ofs.is_open() &&
                g_signal_log_fd >= 0 && g_signal_crash_fd >= 0) {
            return;
        }
        if (g_log_ofs.is_open()) {
            g_log_ofs << current_time_str() << " [JNI] Log reopened with path: " << path << std::endl;
            g_log_ofs.close();
        }
        g_log_path = path;
        if (!g_log_path.empty()) {
            g_log_ofs.open(g_log_path, std::ios::app | std::ios::binary);
            if (g_log_ofs) {
                g_log_ofs << current_time_str() << " [JNI] Log opened: " << g_log_path << std::endl;
                g_log_ofs.flush();
            } else {
                LOGE("Failed to open log file: %s", g_log_path.c_str());
            }
        }

        if (!g_log_path.empty()) {
            const int signal_log_fd = open(g_log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
            if (signal_log_fd >= 0) {
                replace_signal_fd(&g_signal_log_fd, signal_log_fd);
            } else {
                LOGE("Failed to open signal log file: %s", g_log_path.c_str());
                if (g_log_ofs) {
                    g_log_ofs << current_time_str() << " [JNI] WARNING: Failed to open signal log file: "
                              << g_log_path << std::endl;
                    g_log_ofs.flush();
                }
            }

            const std::string crash_path = derive_crash_log_path(g_log_path);
            if (!crash_path.empty()) {
                const int signal_crash_fd = open(crash_path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
                if (signal_crash_fd >= 0) {
                    replace_signal_fd(&g_signal_crash_fd, signal_crash_fd);
                    if (g_log_ofs) {
                        g_log_ofs << current_time_str() << " [JNI] Native crash log path: " << crash_path << std::endl;
                        g_log_ofs.flush();
                    }
                } else {
                    LOGE("Failed to open native crash log file: %s", crash_path.c_str());
                    if (g_log_ofs) {
                        g_log_ofs << current_time_str() << " [JNI] WARNING: Failed to open native crash log file: "
                                  << crash_path << std::endl;
                        g_log_ofs.flush();
                    }
                }
            }

            if (g_log_ofs) {
                g_log_ofs << current_time_str()
                          << " [JNI] Signal crash capture ready: signal_log_fd="
                          << (g_signal_log_fd >= 0 ? "open" : "unavailable")
                          << " signal_crash_fd="
                          << (g_signal_crash_fd >= 0 ? "open" : "unavailable")
                          << std::endl;
                g_log_ofs.flush();
            }
        }
    }
}

// ---------------- JNI: setLogLevel ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setLogLevel(
        JNIEnv *, jobject, jint level) {
    int sanitized = level;
    if (sanitized < GGML_LOG_LEVEL_NONE) sanitized = GGML_LOG_LEVEL_NONE;
    if (sanitized > GGML_LOG_LEVEL_ERROR) sanitized = GGML_LOG_LEVEL_ERROR;
    g_log_level.store(sanitized);
    log_to_file(std::string("log level set to ") + std::to_string(sanitized));
}

extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setDownloadCaBundlePath(
        JNIEnv * env, jobject, jstring jpath) {
    const std::string path = jpath ? jstring_to_std(env, jpath) : "";
    {
        std::lock_guard<std::mutex> lock(g_download_ca_bundle_mutex);
        g_download_ca_bundle_path = path;
    }

    if (path.empty()) {
        log_to_file("download: cleared CA bundle path", GGML_LOG_LEVEL_WARN);
    } else {
        log_to_file(std::string("download: configured CA bundle path=") + path);
    }
}

// ---------------- JNI: download ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_download(
        JNIEnv* env,
        jobject thiz,
        jstring jurl,
        jstring jpath) {
    ensure_fatal_signal_handlers_installed();

    // Ensure we have a stored JavaVM for callbacks even if init() has not been called yet
    if (!g_jvm) {
        if (env->GetJavaVM(&g_jvm) != JNI_OK) {
            g_jvm = nullptr;
            log_to_file("download: GetJavaVM failed", GGML_LOG_LEVEL_WARN);
        } else {
            log_to_file("download: JavaVM stored");
        }
    }

    const char* url_chars  = env->GetStringUTFChars(jurl,  nullptr);
    const char* path_chars = env->GetStringUTFChars(jpath, nullptr);

    if (!url_chars || !path_chars) {
        if (url_chars)  env->ReleaseStringUTFChars(jurl, url_chars);
        if (path_chars) env->ReleaseStringUTFChars(jpath, path_chars);
        log_to_file("download: invalid args", GGML_LOG_LEVEL_ERROR);
        return new_java_string_utf8(env, "invalid args");
    }

    const std::string url(url_chars);
    const std::string path(path_chars);
    env->ReleaseStringUTFChars(jurl, url_chars);
    env->ReleaseStringUTFChars(jpath, path_chars);

    {
        std::ostringstream ss;
        ss << "download: start url=" << url << " path=" << path;
        log_to_file(ss.str());
    }

    ProgressData pd;
    pd.last_percent = -1;
    pd.thiz_global = env->NewGlobalRef(thiz);
    pd.onProgressMethod = nullptr;

    jclass cls = env->GetObjectClass(thiz);
    if (cls) {
        pd.onProgressMethod = env->GetMethodID(cls, "onDownloadProgress", "(I)V");
    }

    std::string error_message;
    const bool ok = download_model_with_splits(url, path, &pd, error_message);
    if (pd.thiz_global) env->DeleteGlobalRef(pd.thiz_global);
    if (!ok) {
        if (error_message.empty()) {
            error_message = "download failed";
        }
        return new_java_string_utf8(env, error_message);
    }

    log_to_file("download: ok");
    return new_java_string_utf8(env, "ok");
}

// ---------------- JNI: init ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_init(
        JNIEnv *env, jobject,
        jstring jModelPath
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_fatal_signal_handlers_installed();

    auto clear_partial_init = []() {
        release_multimodal_locked("init");
        release_context_locked("init");
        release_model_locked("init");
        release_backend_locked("init");
        g_current_mmproj_path.clear();
        g_supports_vision = false;
        g_supports_audio = false;
    };

    try {
        log_to_file("init: start");

        // ★ llama.cpp 内部ログを JNI 側へ流す
        llama_log_set(llama_log_callback, nullptr);
        log_to_file("init: llama_log_callback registered");

        std::string model_path = jstring_to_std(env, jModelPath);

        {
            std::ostringstream ss;
            ss << "init: model_path=" << model_path;
            log_to_file(ss.str());
        }

        // If the same model is already loaded in JNI, skip re-initialization to avoid heavy work
        if (!g_current_model_path.empty() && g_current_model_path == model_path && g_model && g_ctx) {
            std::ostringstream ss;
            ss << "init: model already initialized at path=" << model_path << "; skipping init";
            log_to_file(ss.str());
            return new_java_string_utf8(env, "ok");
        }

        // Defensively free existing resources before loading a new model
        if (g_mtmd) {
            log_to_file("init: freeing existing multimodal context before re-init");
            release_multimodal_locked("init");
        }
        g_current_mmproj_path.clear();
        g_supports_vision = false;
        g_supports_audio = false;
        if (g_ctx) {
            log_to_file("init: freeing existing context before re-init");
            release_context_locked("init");
        }
        if (g_model) {
            log_to_file("init: freeing existing model before re-init");
            release_model_locked("init");
        }
        // The ggml backend is process-wide; keep it initialized across model switches
        // (ensure_backend_initialized_locked is idempotent) instead of churning it each time.
        if (!g_current_model_path.empty()) {
            log_to_file("init: clearing previous model path");
            g_current_model_path.clear();
        }

        {
            std::ifstream ifs(model_path, std::ios::binary | std::ios::ate);
            if (!ifs) {
                std::ostringstream ss;
                ss << "init: model file cannot be opened: " << model_path
                   << " errno=" << errno << " strerror=" << std::strerror(errno);
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, "model file open failed");
            } else {
                auto sz = ifs.tellg();
                std::ostringstream ss;
                ss << "init: model file exists, size=" << sz << " bytes";
                log_to_file(ss.str());
                ifs.close();
            }
        }

        {
            std::string missing_split_path;
            if (find_missing_split_shard_path(model_path, missing_split_path)) {
                std::ostringstream ss;
                ss << "init: missing split shard: " << missing_split_path;
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, ss.str());
            }
        }

        {
            std::ifstream ifh(model_path, std::ios::binary);
            if (ifh) {
                char hdr_buf[64];
                ifh.read(hdr_buf, sizeof(hdr_buf));
                std::streamsize got = ifh.gcount();
                std::ostringstream ss;
                ss << "init: header(" << got << "):";
                ss << std::hex << std::setfill('0');
                for (std::streamsize i = 0; i < got; ++i) {
                    ss << ' ' << std::setw(2)
                       << (static_cast<unsigned int>(static_cast<unsigned char>(hdr_buf[i])));
                }
                ss << " | ";
                for (std::streamsize i = 0; i < got; ++i) {
                    unsigned char c = static_cast<unsigned char>(hdr_buf[i]);
                    ss << (std::isprint(c) ? static_cast<char>(c) : '.');
                }
                log_to_file(ss.str());
            } else {
                std::ostringstream ss;
                ss << "init: header dump failed to open file: " << model_path;
                log_to_file(ss.str(), GGML_LOG_LEVEL_WARN);
            }
        }

        if (env->GetJavaVM(&g_jvm) != JNI_OK) {
            g_jvm = nullptr;
            log_to_file("init: GetJavaVM failed", GGML_LOG_LEVEL_WARN);
        } else {
            log_to_file("init: JavaVM stored");
        }

        ensure_backend_initialized_locked("init");
        log_requested_load_params("init");
        const size_t backend_count = ggml_backend_reg_count();
        if (backend_count == 0) {
            return new_java_string_utf8(env, "no ggml backends registered");
        }

        llama_model_params mparams = build_model_params_locked(false);
        log_model_params("init", mparams);

        {
            using namespace std::chrono;
            auto t0 = high_resolution_clock::now();
            g_model = llama_model_load_from_file(model_path.c_str(), mparams);
            auto t1 = high_resolution_clock::now();
            auto ms = duration_cast<milliseconds>(t1 - t0).count();

            std::ostringstream ss;
            if (!g_model) {
                const std::string hint = build_model_load_failure_message("init");
                ss << hint << " after "
                   << ms << " ms. path_len=" << model_path.size();
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, hint);
            } else {
                ss << "init: model loaded successfully in " << ms << " ms";
                log_to_file(ss.str());
            }
        }
        {
            const bool is_recurrent = llama_model_is_recurrent(g_model);
            const bool is_hybrid = llama_model_is_hybrid(g_model);
            std::ostringstream ss;
            ss << "init: model arch recurrent=" << (is_recurrent ? "true" : "false")
               << " hybrid=" << (is_hybrid ? "true" : "false")
               << " n_batch=" << g_n_batch
               << " n_ubatch=" << g_n_batch;
            log_to_file(ss.str());
        }

        llama_context_params cparams = build_context_params_locked();

        {
            using namespace std::chrono;
            auto t0 = high_resolution_clock::now();
            g_ctx = llama_init_from_model(g_model, cparams);
            auto t1 = high_resolution_clock::now();
            auto ms = duration_cast<milliseconds>(t1 - t0).count();

            std::ostringstream ss;
            if (!g_ctx) {
                const std::string hint = build_context_load_failure_message("init");
                ss << hint << " after "
                   << ms << " ms";
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, hint);
            } else {
                ss << "init: context created successfully in " << ms << " ms";
                log_to_file(ss.str());
            }
        }
        g_current_model_path = model_path;
        g_current_mmproj_path = initialize_optional_multimodal_support_locked(
                model_path,
                "",
                false,
                "init");
        g_current_audio_requested = false;
        maybe_init_speculative_locked("init");
        log_to_file("init: initialization complete");

        return new_java_string_utf8(env, "ok");
    } catch (const std::exception & e) {
        std::ostringstream ss;
        ss << "init: exception: " << e.what();
        log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
        clear_partial_init();
        return new_java_string_utf8(env, "init exception");
    } catch (...) {
        log_to_file("init: unknown native exception", GGML_LOG_LEVEL_ERROR);
        clear_partial_init();
        return new_java_string_utf8(env, "init unknown exception");
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_initWithMmproj(
        JNIEnv *env, jobject,
        jstring jModelPath,
        jstring jMmprojPath,
        jboolean jEnableAudio
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_fatal_signal_handlers_installed();

    auto clear_partial_init = []() {
        release_multimodal_locked("initWithMmproj");
        release_context_locked("initWithMmproj");
        release_model_locked("initWithMmproj");
        release_backend_locked("initWithMmproj");
        g_current_mmproj_path.clear();
        g_supports_vision = false;
        g_supports_audio = false;
        g_current_audio_requested = false;
    };

    try {
        log_to_file("initWithMmproj: start");

        llama_log_set(llama_log_callback, nullptr);
        mtmd_helper_log_set(llama_log_callback, nullptr);
        log_to_file("initWithMmproj: llama/mtmd log callbacks registered");

        std::string model_path = jstring_to_std(env, jModelPath);
        std::string mmproj_path = jMmprojPath ? jstring_to_std(env, jMmprojPath) : "";
        const bool enable_audio = jEnableAudio == JNI_TRUE;

        {
            std::ostringstream ss;
            ss << "initWithMmproj: model_path=" << model_path
               << " mmproj_path=" << (mmproj_path.empty() ? "<none>" : mmproj_path)
               << " enable_audio=" << (enable_audio ? "true" : "false");
            log_to_file(ss.str());
        }

        // Only skip re-init when the *effective* multimodal state already matches the request:
        //  - request has no mmproj  -> skip only if no projector is currently active
        //    (both the configured path is empty AND g_mtmd == nullptr). The previous
        //    "|| g_mtmd != nullptr" term wrongly skipped when a projector was loaded, so
        //    clearing the projector never took effect natively.
        //  - request has an mmproj  -> skip only if the same projector path is loaded.
        if (!g_current_model_path.empty()
                && g_current_model_path == model_path
                && g_model && g_ctx
                && (!enable_audio || g_current_audio_requested)
                && ((mmproj_path.empty() && g_current_mmproj_path.empty() && g_mtmd == nullptr)
                || (!mmproj_path.empty() && g_current_mmproj_path == mmproj_path))) {
            std::ostringstream ss;
            ss << "initWithMmproj: model already initialized at path=" << model_path
               << " mmproj=" << (g_current_mmproj_path.empty() ? "<none>" : g_current_mmproj_path)
               << " audio_requested=" << (g_current_audio_requested ? "true" : "false")
               << "; skipping init";
            log_to_file(ss.str());
            return new_java_string_utf8(env, "ok");
        }

        if (g_mtmd) {
            log_to_file("initWithMmproj: freeing existing multimodal context before re-init");
            release_multimodal_locked("initWithMmproj");
        }
        g_current_mmproj_path.clear();
        g_supports_vision = false;
        g_supports_audio = false;
        g_current_audio_requested = false;
        if (g_ctx) {
            log_to_file("initWithMmproj: freeing existing context before re-init");
            release_context_locked("initWithMmproj");
        }
        if (g_model) {
            log_to_file("initWithMmproj: freeing existing model before re-init");
            release_model_locked("initWithMmproj");
        }
        // The ggml backend is process-wide; keep it initialized across model switches
        // (ensure_backend_initialized_locked is idempotent) instead of churning it each time.
        if (!g_current_model_path.empty()) {
            log_to_file("initWithMmproj: clearing previous model path");
            g_current_model_path.clear();
        }

        {
            std::ifstream ifs(model_path, std::ios::binary | std::ios::ate);
            if (!ifs) {
                std::ostringstream ss;
                ss << "initWithMmproj: model file cannot be opened: " << model_path
                   << " errno=" << errno << " strerror=" << std::strerror(errno);
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, "model file open failed");
            } else {
                auto sz = ifs.tellg();
                std::ostringstream ss;
                ss << "initWithMmproj: model file exists, size=" << sz << " bytes";
                log_to_file(ss.str());
                ifs.close();
            }
        }

        if (!mmproj_path.empty()) {
            std::ifstream ifs(mmproj_path, std::ios::binary | std::ios::ate);
            if (!ifs) {
                std::ostringstream ss;
                ss << "initWithMmproj: mmproj file cannot be opened: " << mmproj_path
                   << " errno=" << errno << " strerror=" << std::strerror(errno);
                log_to_file(ss.str(), GGML_LOG_LEVEL_WARN);
                mmproj_path.clear();
            } else {
                auto sz = ifs.tellg();
                std::ostringstream ss;
                ss << "initWithMmproj: mmproj file exists, size=" << sz << " bytes";
                log_to_file(ss.str());
                ifs.close();
            }
        }

        {
            std::string missing_split_path;
            if (find_missing_split_shard_path(model_path, missing_split_path)) {
                std::ostringstream ss;
                ss << "initWithMmproj: missing split shard: " << missing_split_path;
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, ss.str());
            }
        }

        if (env->GetJavaVM(&g_jvm) != JNI_OK) {
            g_jvm = nullptr;
            log_to_file("initWithMmproj: GetJavaVM failed", GGML_LOG_LEVEL_WARN);
        } else {
            log_to_file("initWithMmproj: JavaVM stored");
        }

        ensure_backend_initialized_locked("initWithMmproj");
        log_requested_load_params("initWithMmproj");
        const size_t backend_count = ggml_backend_reg_count();
        if (backend_count == 0) {
            return new_java_string_utf8(env, "no ggml backends registered");
        }

        llama_model_params mparams = build_model_params_locked(!mmproj_path.empty());
        log_model_params("initWithMmproj", mparams);

        {
            using namespace std::chrono;
            auto t0 = high_resolution_clock::now();
            g_model = llama_model_load_from_file(model_path.c_str(), mparams);
            auto t1 = high_resolution_clock::now();
            auto ms = duration_cast<milliseconds>(t1 - t0).count();

            std::ostringstream ss;
            if (!g_model) {
                const std::string hint = build_model_load_failure_message("initWithMmproj");
                ss << hint << " after "
                   << ms << " ms. path_len=" << model_path.size();
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, hint);
            } else {
                ss << "initWithMmproj: model loaded successfully in " << ms << " ms";
                log_to_file(ss.str());
            }
        }
        {
            const bool is_recurrent = llama_model_is_recurrent(g_model);
            const bool is_hybrid = llama_model_is_hybrid(g_model);
            std::ostringstream ss;
            ss << "initWithMmproj: model arch recurrent=" << (is_recurrent ? "true" : "false")
               << " hybrid=" << (is_hybrid ? "true" : "false")
               << " n_batch=" << g_n_batch
               << " n_ubatch=" << g_n_batch;
            log_to_file(ss.str());
        }

        llama_context_params cparams = build_context_params_locked();

        {
            using namespace std::chrono;
            auto t0 = high_resolution_clock::now();
            g_ctx = llama_init_from_model(g_model, cparams);
            auto t1 = high_resolution_clock::now();
            auto ms = duration_cast<milliseconds>(t1 - t0).count();

            std::ostringstream ss;
            if (!g_ctx) {
                const std::string hint = build_context_load_failure_message("initWithMmproj");
                ss << hint << " after "
                   << ms << " ms";
                log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
                return new_java_string_utf8(env, hint);
            } else {
                ss << "initWithMmproj: context created successfully in " << ms << " ms";
                log_to_file(ss.str());
            }
        }
        const std::string selected_mmproj_path = initialize_optional_multimodal_support_locked(
                model_path,
                mmproj_path,
                enable_audio,
                "initWithMmproj");

        g_current_model_path = model_path;
        g_current_mmproj_path = selected_mmproj_path;
        g_current_audio_requested = enable_audio && !selected_mmproj_path.empty();
        maybe_init_speculative_locked("initWithMmproj");
        log_to_file("initWithMmproj: initialization complete");

        return new_java_string_utf8(env, "ok");
    } catch (const std::exception & e) {
        std::ostringstream ss;
        ss << "initWithMmproj: exception: " << e.what();
        log_to_file(ss.str(), GGML_LOG_LEVEL_ERROR);
        clear_partial_init();
        return new_java_string_utf8(env, "init exception");
    } catch (...) {
        log_to_file("initWithMmproj: unknown native exception", GGML_LOG_LEVEL_ERROR);
        clear_partial_init();
        return new_java_string_utf8(env, "init unknown exception");
    }
}

// Cheap metadata-only check for whether an mmproj advertises an audio encoder.
// Returns 1 = has audio, 0 = no audio (vision-only), -1 = unknown (couldn't read).
static int mmproj_audio_capability(const std::string & mmproj_path) {
    struct gguf_init_params p = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(mmproj_path.c_str(), p);
    if (!g) {
        return -1;
    }
    const bool has_audio = gguf_kv_is_true(g, "clip.has_audio_encoder");
    gguf_free(g);
    return has_audio ? 1 : 0;
}

// ---------------- JNI: mmprojSupportsAudio ----------------
// Lets the Java layer request the audio encoder only for mmproj that actually have one, so a
// vision-only mmproj is not loaded with enable_audio=true (which would keep the reuse fast-path's
// audio clause unsatisfied and force a full model reload before every generation).
extern "C"
JNIEXPORT jint JNICALL
Java_com_micklab_llama_LlamaNative_mmprojSupportsAudio(
        JNIEnv * env,
        jobject,
        jstring jMmprojPath) {
    try {
        const std::string mmproj_path = jMmprojPath ? jstring_to_std(env, jMmprojPath) : "";
        if (mmproj_path.empty()) {
            return -1;
        }
        return (jint) mmproj_audio_capability(mmproj_path);
    } catch (...) {
        return -1;
    }
}

// ---------------- JNI: validateMmproj ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_validateMmproj(
        JNIEnv * env,
        jobject,
        jstring jModelPath,
        jstring jMmprojPath) {
    std::string result = "unknown";
    try {
        const std::string model_path  = jModelPath  ? jstring_to_std(env, jModelPath)  : "";
        const std::string mmproj_path = jMmprojPath ? jstring_to_std(env, jMmprojPath) : "";
        if (!model_path.empty() && !mmproj_path.empty()) {
            result = validate_mmproj_for_model(model_path, mmproj_path);
        }
    } catch (const std::exception & e) {
        log_to_file(std::string("validateMmproj: exception: ") + e.what(), GGML_LOG_LEVEL_ERROR);
        result = "unknown";
    } catch (...) {
        log_to_file("validateMmproj: unknown exception", GGML_LOG_LEVEL_ERROR);
        result = "unknown";
    }
    return new_java_string_utf8(env, result);
}

// ---------------- JNI: setLoadParameters ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setLoadParameters(
        JNIEnv *, jobject,
        jint nCtx, jint nThreads, jint nBatch,
        jfloat temp, jfloat topP, jint topK, jint nGpuLayers
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    g_n_ctx = nCtx > 0 ? nCtx : 1;
    g_n_threads = nThreads > 0 ? nThreads : 1;
    g_n_batch = nBatch > 0 ? nBatch : 1;
    g_temp = temp;
    g_top_p = topP;
    g_top_k = topK;
    g_n_gpu_layers = nGpuLayers;

    {
        std::ostringstream ss;
        ss << "setLoadParameters: n_ctx=" << g_n_ctx
           << " n_threads=" << g_n_threads
           << " n_batch=" << g_n_batch
           << " temp=" << g_temp
           << " top_p=" << g_top_p
           << " top_k=" << g_top_k
           << " n_gpu_layers=" << g_n_gpu_layers;
        log_to_file(ss.str());
    }
}

// ---------------- JNI: setUseMmap ----------------
// Records whether the model file should be memory-mapped at load. Applied at the next
// init/initWithMmproj (build_model_params_locked reads g_use_mmap). Disabling avoids the
// large contiguous virtual-address reservation mmap needs.
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setUseMmap(
        JNIEnv *, jobject, jboolean useMmap
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_use_mmap = (useMmap == JNI_TRUE);
    std::ostringstream ss;
    ss << "setUseMmap: use_mmap=" << (g_use_mmap ? "true" : "false");
    log_to_file(ss.str());
}

// ---------------- JNI: applyLoraAdapter ----------------
// 現在ロード中のモデル/コンテキストへ LoRA アダプタGGUF を適用する。
// 既に別アダプタ適用中なら差し替える。成功="", 失敗=エラーメッセージ。
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_applyLoraAdapter(
        JNIEnv *env, jobject, jstring jPath, jfloat scale
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        return env->NewStringUTF("モデル未ロードです。先にモデルを読み込んでください。");
    }
    std::string path = jPath ? jstring_to_std(env, jPath) : "";
    if (path.empty()) {
        return env->NewStringUTF("アダプタのパスが空です。");
    }
    // 既存アダプタを外して解放（差し替え）
    if (g_lora) {
        llama_set_adapters_lora(g_ctx, nullptr, 0, nullptr);
        llama_adapter_lora_free(g_lora);
        g_lora = nullptr;
        g_lora_path.clear();
    }
    llama_adapter_lora *ad = llama_adapter_lora_init(g_model, path.c_str());
    if (!ad) {
        return env->NewStringUTF("アダプタ読込失敗（GGUF形式/対象アーキ不一致の可能性）。");
    }
    llama_adapter_lora *arr[1]   = { ad };
    float               scl[1]   = { (float) scale };
    int32_t rc = llama_set_adapters_lora(g_ctx, arr, 1, scl);
    if (rc != 0) {
        llama_adapter_lora_free(ad);
        return env->NewStringUTF("アダプタ適用失敗（llama_set_adapters_lora）。");
    }
    g_lora = ad;
    g_lora_path = path;
    g_lora_scale = (float) scale;
    std::ostringstream ss;
    ss << "applyLoraAdapter: path=" << path << " scale=" << g_lora_scale;
    log_to_file(ss.str());
    return env->NewStringUTF("");
}

// ---------------- JNI: clearLoraAdapter ----------------
// 適用中の LoRA アダプタを外して解放する。未適用なら何もしない。
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_clearLoraAdapter(
        JNIEnv *, jobject
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_lora) {
        return;
    }
    if (g_ctx) {
        llama_set_adapters_lora(g_ctx, nullptr, 0, nullptr);
    }
    llama_adapter_lora_free(g_lora);
    g_lora = nullptr;
    g_lora_path.clear();
    log_to_file("clearLoraAdapter: adapter removed");
}

// ---------------- JNI: getLoraAdapterInfo ----------------
// 適用中アダプタの "path\tscale" を返す。未適用は ""（UI表示用）。
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_getLoraAdapterInfo(
        JNIEnv *env, jobject
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_lora) {
        return env->NewStringUTF("");
    }
    std::ostringstream ss;
    ss << g_lora_path << "\t" << g_lora_scale;
    return env->NewStringUTF(ss.str().c_str());
}

// ---------------- JNI: setSpeculative (MTP draft-mtp) ----------------
// Records the MTP draft-model path + settings; applied at the next init/initWithMmproj
// (maybe_init_speculative_locked). enabled=false or empty path => plain decoding.
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setSpeculative(
        JNIEnv *env, jobject,
        jstring jMtpModelPath, jint nDraft, jboolean enabled
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_mtp_path    = jMtpModelPath ? jstring_to_std(env, jMtpModelPath) : "";
    g_mtp_n_draft = nDraft > 0 ? nDraft : 2;
    // Empty path is valid: it means "use the loaded model's own embedded MTP head".
    g_mtp_enabled = (enabled == JNI_TRUE);
    std::ostringstream ss;
    ss << "setSpeculative: enabled=" << (g_mtp_enabled ? "true" : "false")
       << " n_draft=" << g_mtp_n_draft << " mtp=" << g_mtp_path
       << " (applies at next init)";
    log_to_file(ss.str());
}

// ---------------- JNI: setParameters ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setParameters(
        JNIEnv *env, jobject,
        jint penaltyLastN, jfloat penaltyRepeat, jfloat penaltyFreq, jfloat penaltyPresent,
        jint mirostat, jfloat mirostatTau, jfloat mirostatEta,
        jfloat minP, jfloat typicalP,
        jfloat dynatempRange, jfloat dynatempExponent,
        jfloat xtcProbability, jfloat xtcThreshold,
        jfloat topNSigma,
        jfloat dryMultiplier, jfloat dryBase, jint dryAllowedLength, jint dryPenaltyLastN,
        jstring jDrySequenceBreakers
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    // Penalty parameters
    g_penalty_last_n = penaltyLastN;
    g_penalty_repeat = penaltyRepeat;
    g_penalty_freq = penaltyFreq;
    g_penalty_present = penaltyPresent;
    
    // Mirostat parameters
    g_mirostat = mirostat;
    g_mirostat_tau = mirostatTau;
    g_mirostat_eta = mirostatEta;
    
    // Additional sampler parameters
    g_min_p = minP;
    g_typical_p = typicalP;
    g_dynatemp_range = dynatempRange;
    g_dynatemp_exponent = dynatempExponent;
    g_xtc_probability = xtcProbability;
    g_xtc_threshold = xtcThreshold;
    g_top_n_sigma = topNSigma;
    
    // DRY parameters
    g_dry_multiplier = dryMultiplier;
    g_dry_base = dryBase;
    g_dry_allowed_length = dryAllowedLength;
    g_dry_penalty_last_n = dryPenaltyLastN;
    g_dry_sequence_breakers = jstring_to_std(env, jDrySequenceBreakers);
    
    {
        std::ostringstream ss;
        ss << "setParameters: penalty_last_n=" << g_penalty_last_n
           << " penalty_repeat=" << g_penalty_repeat
           << " penalty_freq=" << g_penalty_freq
           << " penalty_present=" << g_penalty_present
           << " mirostat=" << g_mirostat
           << " mirostat_tau=" << g_mirostat_tau
           << " mirostat_eta=" << g_mirostat_eta
           << " min_p=" << g_min_p
           << " typical_p=" << g_typical_p
           << " dynatemp_range=" << g_dynatemp_range
           << " dynatemp_exponent=" << g_dynatemp_exponent
           << " xtc_probability=" << g_xtc_probability
           << " xtc_threshold=" << g_xtc_threshold
           << " top_n_sigma=" << g_top_n_sigma
           << " dry_multiplier=" << g_dry_multiplier
           << " dry_base=" << g_dry_base
           << " dry_allowed_length=" << g_dry_allowed_length
           << " dry_penalty_last_n=" << g_dry_penalty_last_n
           << " dry_sequence_breakers=\"" << g_dry_sequence_breakers << "\"";
        log_to_file(ss.str());
    }
}

// ---------------- JNI: setTokenListener ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setTokenListener(
        JNIEnv *env, jobject /*thiz*/, jobject listener) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // Ensure JavaVM stored
    if (!g_jvm) {
        if (env->GetJavaVM(&g_jvm) != JNI_OK) {
            g_jvm = nullptr;
            log_to_file("setTokenListener: GetJavaVM failed", GGML_LOG_LEVEL_WARN);
        } else {
            log_to_file("setTokenListener: JavaVM stored");
        }
    }
    // Delete previous global ref if exists
    if (g_token_listener) {
        env->DeleteGlobalRef(g_token_listener);
        g_token_listener = nullptr;
        g_token_onToken = nullptr;
        g_token_onComplete = nullptr;
        g_token_onError = nullptr;
    }
    if (listener == nullptr) {
        log_to_file("setTokenListener: cleared listener");
        return;
    }
    g_token_listener = env->NewGlobalRef(listener);
    jclass cls = env->GetObjectClass(listener);
    if (cls) {
        g_token_onToken = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");
        g_token_onComplete = env->GetMethodID(cls, "onComplete", "()V");
        g_token_onError = env->GetMethodID(cls, "onError", "(Ljava/lang/String;)V");
    }
    log_to_file("setTokenListener: listener registered");
    if (should_log_debug()) {
        LOGD("setTokenListener: listener registered");
    }
}

// ---------------- JNI: setSamplerOrder ----------------
// Sets the sampler chain order for the NEXT generate() call only.
// The order is a semicolon-delimited list of sampler names recognised by the WebUI,
// e.g. "top_k;top_p;min_p;temperature".  An empty string resets to the built-in default
// (top_k → typ_p → top_p → min_p → temperature).
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setSamplerOrder(
        JNIEnv *env, jobject, jstring jorder) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_sampler_order = jorder != nullptr ? jstring_to_std(env, jorder) : "";
    if (should_log_debug()) {
        LOGD("setSamplerOrder: order='%s'", g_sampler_order.c_str());
    }
}

// ---------------- JNI: setGrammar ----------------
// Sets the grammar constraint applied by the next generate() call.
// Precedence: a non-empty GBNF string is used as-is (TYPE_USER); otherwise a non-empty JSON
// Schema is converted to GBNF via json_schema_to_grammar (TYPE_OUTPUT_FORMAT, same as arg.cpp);
// otherwise the constraint is cleared. OllamaApiServer calls this from the /api/chat and
// /api/generate "format"/"grammar" fields.
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setGrammar(
        JNIEnv * env, jobject, jstring jgbnf, jstring jschema) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string gbnf;
    std::string schema;
    if (jgbnf != nullptr) {
        const char * c = env->GetStringUTFChars(jgbnf, nullptr);
        if (c) { gbnf = c; env->ReleaseStringUTFChars(jgbnf, c); }
    }
    if (jschema != nullptr) {
        const char * c = env->GetStringUTFChars(jschema, nullptr);
        if (c) { schema = c; env->ReleaseStringUTFChars(jschema, c); }
    }
    if (!gbnf.empty()) {
        g_grammar = { COMMON_GRAMMAR_TYPE_USER, gbnf };
    } else if (!schema.empty()) {
        // Convert the JSON schema to GBNF now, the same way arg.cpp does it.
        // common_sampler_init passes grammar.grammar directly to llama_sampler_init_grammar,
        // which expects GBNF — storing the raw JSON schema there causes "failed to parse grammar".
        try {
            const std::string gbnf_from_schema = json_schema_to_grammar(common_json::parse(schema));
            g_grammar = { COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT, gbnf_from_schema };
        } catch (const std::exception & e) {
            log_to_file(std::string("setGrammar: json_schema_to_grammar failed: ") + e.what(),
                        GGML_LOG_LEVEL_ERROR);
            g_grammar = {};
        }
    } else {
        g_grammar = {};
    }
    log_to_file(std::string("setGrammar: grammar ")
                + (g_grammar.empty() ? "cleared"
                                     : "set (" + std::to_string(g_grammar.grammar.size()) + " bytes)"));
}

// ---------------- JNI: embed ----------------
// Computes a sentence embedding for `text` with the loaded model and returns JSON
// {"embedding":[...]} (or {"error":"..."}). For generative (causal) models the per-token
// hidden states are mean-pooled (pooling_type NONE) and L2-normalized; a dedicated embedding
// model with mean/cls pooling yields better quality. The KV cache is cleared before and after
// so an in-progress chat session is not polluted.
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_embed(
        JNIEnv * env, jobject, jstring jtext) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        return new_java_string_utf8(env, "{\"error\":\"model not loaded\"}");
    }
    std::string text;
    if (jtext != nullptr) {
        const char * c = env->GetStringUTFChars(jtext, nullptr);
        if (c) { text = c; env->ReleaseStringUTFChars(jtext, c); }
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> tokens = common_tokenize(vocab, text, true, false);
    if (tokens.empty()) {
        return new_java_string_utf8(env, "{\"error\":\"empty input\"}");
    }
    if ((int) tokens.size() > g_n_ctx) {
        tokens.resize(g_n_ctx);
    }
    // llama_decode asserts n_tokens <= n_batch; truncate so long texts don't SIGABRT.
    // Mean-pooling over the first n_batch tokens is acceptable for embedding quality.
    if ((int) tokens.size() > g_n_batch) {
        tokens.resize(g_n_batch);
    }

    const int n_embd = llama_model_n_embd_out(g_model);
    if (n_embd <= 0) {
        return new_java_string_utf8(env, "{\"error\":\"model has no embedding output\"}");
    }

    reusable_token_batch batch((int32_t) tokens.size());
    if (!batch.valid()) {
        return new_java_string_utf8(env, "{\"error\":\"batch alloc failed\"}");
    }
    batch.batch.n_tokens = (int32_t) tokens.size();
    for (int i = 0; i < (int) tokens.size(); ++i) {
        // logits=true => per-token embeddings are produced (required for pooling_type NONE)
        batch.set_token(i, tokens[i], (llama_pos) i, true);
    }

    llama_memory_t mem = llama_get_memory(g_ctx);
    llama_memory_clear(mem, true);
    llama_set_embeddings(g_ctx, true);

    const int rc = llama_decode(g_ctx, batch.batch);

    std::vector<float> pooled(n_embd, 0.0f);
    bool ok = (rc >= 0);
    if (ok) {
        const enum llama_pooling_type pooling = llama_pooling_type(g_ctx);
        if (pooling == LLAMA_POOLING_TYPE_NONE) {
            int counted = 0;
            for (int i = 0; i < (int) tokens.size(); ++i) {
                const float * e = llama_get_embeddings_ith(g_ctx, i);
                if (e == nullptr) continue;
                for (int d = 0; d < n_embd; ++d) pooled[d] += e[d];
                ++counted;
            }
            if (counted > 0) {
                for (int d = 0; d < n_embd; ++d) pooled[d] /= (float) counted;
            } else {
                ok = false;
            }
        } else {
            const float * e = llama_get_embeddings_seq(g_ctx, 0);
            if (e != nullptr) {
                for (int d = 0; d < n_embd; ++d) pooled[d] = e[d];
            } else {
                ok = false;
            }
        }
    }

    // restore generation mode and drop the scratch tokens from the KV cache
    llama_set_embeddings(g_ctx, false);
    llama_memory_clear(mem, true);

    if (!ok) {
        return new_java_string_utf8(env, "{\"error\":\"embedding decode failed\"}");
    }

    std::vector<float> normed(n_embd, 0.0f);
    common_embd_normalize(pooled.data(), normed.data(), n_embd, 2 /* L2 */);

    std::string out;
    out.reserve((size_t) n_embd * 12 + 32);
    out += "{\"embedding\":[";
    for (int d = 0; d < n_embd; ++d) {
        if (d > 0) out += ',';
        char buf[32];
        snprintf(buf, sizeof(buf), "%.7g", normed[d]);
        out += buf;
    }
    out += "]}";
    return new_java_string_utf8(env, out);
}

// ---------------- JNI: cancelGeneration ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_cancelGeneration(
        JNIEnv *, jobject) {
    g_cancel_generation.store(true);
    log_to_file("cancelGeneration: cancellation requested", GGML_LOG_LEVEL_WARN);
}

static void notify_token_error(const char * errmsg) {
    if (!errmsg || !g_jvm || !g_token_listener || !g_token_onError) {
        return;
    }
    JNIEnv * env = nullptr;
    bool attached = false;
    if (g_jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            attached = true;
        } else {
            env = nullptr;
        }
    }
    if (env) {
        jstring jerr = new_java_string_utf8(env, errmsg);
        if (!jerr) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            jerr = new_java_string_utf8(env, "unknown error");
        }
        if (jerr) {
            env->CallVoidMethod(g_token_listener, g_token_onError, jerr);
            if (env->ExceptionCheck()) env->ExceptionClear();
            env->DeleteLocalRef(jerr);
        }
    }
    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

static void notify_token_delta(const std::string & delta) {
    if (delta.empty() || !g_jvm || !g_token_listener || !g_token_onToken) {
        return;
    }
    JNIEnv * env = nullptr;
    bool attached = false;
    if (g_jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            attached = true;
        } else {
            env = nullptr;
        }
    }
    if (env) {
        jstring jdelta = new_java_string_utf8(env, delta);
        if (jdelta) {
            env->CallVoidMethod(g_token_listener, g_token_onToken, jdelta);
            if (env->ExceptionCheck()) env->ExceptionClear();
            env->DeleteLocalRef(jdelta);
        } else if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }
    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

static void notify_token_complete() {
    if (!g_jvm || !g_token_listener || !g_token_onComplete) {
        return;
    }
    JNIEnv * env = nullptr;
    bool attached = false;
    if (g_jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
            attached = true;
        } else {
            env = nullptr;
        }
    }
    if (env) {
        env->CallVoidMethod(g_token_listener, g_token_onComplete);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

// Last generation decode throughput (tok/s), published for the UI via
// getLastGenerationSpeed(). Atomic so the JNI thread can read it without the mutex.
static std::atomic<double>  g_last_gen_tps{0.0};
// Performance metrics from the most recent generation, exposed via getLastGenerationMetrics().
static std::atomic<int32_t> g_last_n_prompt{0};
static std::atomic<int32_t> g_last_n_eval{0};
static std::atomic<double>  g_last_t_total_ms{0.0};
// Prompt-eval (prefill) and model-load durations of the most recent generation,
// in milliseconds. Exposed so the API layer can report Ollama-style ns durations.
static std::atomic<double>  g_last_t_p_eval_ms{0.0};
static std::atomic<double>  g_last_t_load_ms{0.0};

// Log generation / prompt throughput (tok/s) from the per-generation perf
// counters (reset at the start of each generation). Surfaces in the diagnostics
// log so users can see decode speed per request.
static void log_generation_perf(const char * log_prefix) {
    if (!g_ctx) {
        return;
    }
    const llama_perf_context_data p = llama_perf_context(g_ctx);
    const double gen_tps = (p.t_eval_ms   > 0.0) ? (1.0e3 * p.n_eval   / p.t_eval_ms)   : 0.0;
    g_last_gen_tps.store(gen_tps, std::memory_order_relaxed);
    g_last_n_prompt.store(static_cast<int32_t>(p.n_p_eval), std::memory_order_relaxed);
    g_last_n_eval.store(static_cast<int32_t>(p.n_eval),     std::memory_order_relaxed);
    g_last_t_total_ms.store(p.t_eval_ms,                    std::memory_order_relaxed);
    g_last_t_p_eval_ms.store(p.t_p_eval_ms,                 std::memory_order_relaxed);
    g_last_t_load_ms.store(p.t_load_ms,                     std::memory_order_relaxed);
    const double pp_tps  = (p.t_p_eval_ms > 0.0) ? (1.0e3 * p.n_p_eval / p.t_p_eval_ms) : 0.0;
    std::ostringstream ss;
    ss << log_prefix << ": speed: gen " << std::fixed << std::setprecision(2) << gen_tps
       << " tok/s (" << p.n_eval << " tok / " << (p.t_eval_ms / 1.0e3) << " s)"
       << ", prompt " << pp_tps << " tok/s (" << p.n_p_eval << " tok / " << (p.t_p_eval_ms / 1.0e3) << " s)";
    log_to_file(ss.str());
}

// Decide the add_special flag for llama_tokenize. Normally true so the model's
// add_bos_token/add_eos_token policy is honored (see the missing-BOS fix). But some GGUFs are
// mis-tagged with a *normal* token as BOS (e.g. Qwen3.5-2B reports BOS = 11 ','); letting llama
// prepend that corrupts the prompt (an out-of-distribution leading comma). In that specific case
// — the model wants a BOS yet its BOS is not a control token — suppress add_special.
static bool tokenize_add_special_flag(const llama_vocab * vocab) {
    if (llama_vocab_get_add_bos(vocab)) {
        const llama_token bos = llama_vocab_bos(vocab);
        if (bos != LLAMA_TOKEN_NULL &&
            (llama_vocab_get_attr(vocab, bos) & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
            log_to_file("tokenize: model BOS token is not a control token (mis-tagged GGUF);"
                        " suppressing add_special to avoid a spurious leading BOS",
                        GGML_LOG_LEVEL_INFO);
            return false;
        }
    }
    return true;
}

static bool prefill_text_prompt_locked(
        const llama_vocab * vocab,
        const std::string & prompt,
        size_t & n_prompt_tokens,
        llama_pos & n_past,
        std::string & error_message) {
    std::vector<llama_token> tokens(g_n_ctx);
    int32_t n_tokens = llama_tokenize(
            vocab,
            prompt.c_str(),
            static_cast<int>(prompt.size()),
            tokens.data(),
            static_cast<int>(tokens.size()),
            tokenize_add_special_flag(vocab),   // add_special: honor add_bos/eos, but guard mis-tagged BOS
            true
    );

    if (n_tokens < 0) {
        // llama_tokenize returns the negated required count when the buffer (n_ctx) is too small.
        // Emit the unified CTX_TOO_SMALL form so the Java layer can auto-promote n_ctx / suppress
        // the intermediate error from the stream (same handling as the multimodal path).
        error_message = "CTX_TOO_SMALL need=" + std::to_string(-n_tokens)
                      + " have=" + std::to_string(g_n_ctx)
                      + " (prompt too long for context; increase Context Size or shorten the input / reduce MCP tools+history)";
        return false;
    }
    if (n_tokens == 0) {
        error_message = "tokenize produced no tokens (empty prompt)";
        return false;
    }
    if (n_tokens >= g_n_ctx) {
        error_message = "CTX_TOO_SMALL need=" + std::to_string(n_tokens + 16)
                      + " have=" + std::to_string(g_n_ctx)
                      + " (prompt leaves no room in context; increase Context Size)";
        return false;
    }

    tokens.resize(n_tokens);
    n_prompt_tokens = static_cast<size_t>(n_tokens);
    n_past = n_tokens;
    log_to_file("generate: prefill start n_prompt_tokens=" + std::to_string(n_tokens)
                + " n_batch=" + std::to_string(g_n_batch));

    reusable_token_batch prompt_batch(std::min(g_n_batch, n_tokens));
    if (!prompt_batch.valid()) {
        error_message = "failed to allocate prompt batch";
        return false;
    }

    for (int i = 0; i < n_tokens; i += g_n_batch) {
        if (g_cancel_generation.load()) {
            error_message = "generation cancelled";
            return false;
        }

        const int batch_size = std::min(g_n_batch, n_tokens - i);
        prompt_batch.batch.n_tokens = batch_size;

        for (int j = 0; j < batch_size; ++j) {
            prompt_batch.set_token(j, tokens[i + j], i + j, i + j == n_tokens - 1);
        }

        const int rc = llama_decode(g_ctx, prompt_batch.batch);
        if (rc != 0) {
            error_message = "decode failed (prompt)";
            return false;
        }
        // MTP: feed every prompt ubatch to the speculator so its head accumulates the
        // target hidden states it drafts from (no-op when MTP is disabled).
        if (g_spec) { common_speculative_process(g_spec, prompt_batch.batch); }
    }

    log_to_file("generate: prefill complete n_prompt_tokens=" + std::to_string(n_tokens));
    return true;
}

// Prefix KV-cache reuse is enabled only for the text path on CPU/GPU. The Hexagon (NPU)
// KV path is newer/less-tested, so we keep the full-reset behavior there.
static bool kv_prefix_cache_allowed_locked() {
    const bool npu_active = g_npu_enabled && (g_backend_type == 2 || g_backend_type == 3);
    return !npu_active;
}

// Cache-aware text prefill: keep the KV for the longest common prefix with the previous
// prompt and decode only the new suffix (big win for multi-turn MCP). On any mismatch it
// clears the KV and prefills from scratch. Maintains g_kv_cached_tokens == KV contents.
static bool prefill_text_prompt_cached_locked(
        const llama_vocab * vocab,
        const std::string & prompt,
        size_t & n_prompt_tokens,
        llama_pos & n_past,
        std::string & error_message) {
    std::vector<llama_token> tokens(g_n_ctx);
    int32_t n_tokens = llama_tokenize(
            vocab, prompt.c_str(), static_cast<int>(prompt.size()),
            tokens.data(), static_cast<int>(tokens.size()),
            tokenize_add_special_flag(vocab), true);  // add_special: honor add_bos/eos, but guard mis-tagged BOS
    if (n_tokens < 0) {
        error_message = "CTX_TOO_SMALL need=" + std::to_string(-n_tokens)
                      + " have=" + std::to_string(g_n_ctx)
                      + " (prompt too long for context; increase Context Size or shorten the input / reduce MCP tools+history)";
        return false;
    }
    if (n_tokens == 0)        { error_message = "tokenize produced no tokens (empty prompt)"; return false; }
    if (n_tokens >= g_n_ctx)  {
        error_message = "CTX_TOO_SMALL need=" + std::to_string(n_tokens + 16)
                      + " have=" + std::to_string(g_n_ctx)
                      + " (prompt leaves no room in context; increase Context Size)";
        return false;
    }
    tokens.resize(n_tokens);

    llama_memory_t memory = llama_get_memory(g_ctx);
    if (!memory) { error_message = "generation memory unavailable"; return false; }

    // longest common prefix between the cached tokens (already in KV) and the new prompt
    size_t n_match = 0;
    const size_t max_match = std::min(g_kv_cached_tokens.size(), static_cast<size_t>(n_tokens));
    while (n_match < max_match && g_kv_cached_tokens[n_match] == tokens[n_match]) {
        ++n_match;
    }
    // always re-decode at least the final token so fresh logits are available for sampling
    if (n_match > static_cast<size_t>(n_tokens - 1)) {
        n_match = static_cast<size_t>(n_tokens - 1);
    }

    if (n_match == 0) {
        llama_memory_clear(memory, true);
    } else if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(n_match), -1)) {
        // couldn't trim the KV cleanly -> reset and re-prefill the whole prompt
        llama_memory_clear(memory, true);
        n_match = 0;
    }
    llama_perf_context_reset(g_ctx);

    log_to_file("generateOpenAiChatCompletion: prefill (cached) n_tokens=" + std::to_string(n_tokens)
                + " reused=" + std::to_string(n_match));

    reusable_token_batch prompt_batch(std::min(g_n_batch, n_tokens));
    if (!prompt_batch.valid()) { error_message = "failed to allocate prompt batch"; return false; }

    for (int i = static_cast<int>(n_match); i < n_tokens; i += g_n_batch) {
        if (g_cancel_generation.load()) {
            g_kv_cached_tokens.clear();
            error_message = "generation cancelled";
            return false;
        }
        const int batch_size = std::min(g_n_batch, n_tokens - i);
        prompt_batch.batch.n_tokens = batch_size;
        for (int j = 0; j < batch_size; ++j) {
            prompt_batch.set_token(j, tokens[i + j], i + j, i + j == n_tokens - 1);
        }
        if (llama_decode(g_ctx, prompt_batch.batch) != 0) {
            g_kv_cached_tokens.clear();
            error_message = "decode failed (prompt)";
            return false;
        }
    }

    n_prompt_tokens = static_cast<size_t>(n_tokens);
    n_past = n_tokens;
    g_kv_cached_tokens.assign(tokens.begin(), tokens.end());
    return true;
}

static bool prefill_multimodal_prompt_locked(
        const std::string & prompt,
        const std::vector<std::vector<uint8_t>> & media_files,
        size_t & n_prompt_tokens,
        llama_pos & n_past,
        std::string & error_message) {
    if (!g_mtmd) {
        error_message = "multimodal projector not initialized";
        return false;
    }

    std::vector<mtmd_bitmap *> bitmaps;
    bitmaps.reserve(media_files.size());

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        error_message = "failed to allocate multimodal chunks";
        return false;
    }

    auto cleanup = [&]() {
        for (mtmd_bitmap * bitmap : bitmaps) {
            if (bitmap) {
                mtmd_bitmap_free(bitmap);
            }
        }
        mtmd_input_chunks_free(chunks);
    };

    for (const auto & file : media_files) {
        // b10091: mtmd_helper_bitmap_init_from_buf now returns a wrapper (bitmap +
        // optional video_ctx) and takes a `placeholder` flag. We only decode real
        // image/audio buffers here (video is not fed through this path), so pass
        // placeholder=false and take the bitmap; video_ctx is null for these inputs.
        mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_buf(g_mtmd, file.data(), file.size(), false).bitmap;
        if (!bitmap) {
            error_message = "failed to decode multimodal input";
            cleanup();
            return false;
        }
        bitmaps.push_back(bitmap);
    }

    std::vector<const mtmd_bitmap *> bitmap_ptrs(bitmaps.begin(), bitmaps.end());
    // b10091 inserted `size_t text_len` as the 2nd field of mtmd_input_text (used
    // directly as the string length in mtmd_tokenize). Initialise all four fields
    // explicitly so the length/flags are not silently misassigned.
    mtmd_input_text input_text = {
            /* text          */ prompt.c_str(),
            /* text_len      */ prompt.size(),
            /* add_special   */ true,
            /* parse_special */ true,
    };

    const int32_t tokenized = mtmd_tokenize(
            g_mtmd,
            chunks,
            &input_text,
            bitmap_ptrs.data(),
            bitmap_ptrs.size()
    );
    if (tokenized != 0) {
        error_message = "failed to tokenize multimodal prompt";
        cleanup();
        return false;
    }

    n_prompt_tokens = mtmd_helper_get_n_tokens(chunks);

    // Context-fit pre-check (Q1). Qwen3-VL and other native-dynamic-resolution VLMs emit far more
    // image tokens than older models (e.g. ~3200 for one image), so the full multimodal prompt can
    // exceed n_ctx. Without this guard mtmd_helper_eval_chunks() below fails deep in the decode loop
    // ("failed to find a memory slot for batch of size N") with a cryptic, unactionable error. Catch
    // it here — before eval — and emit a structured, machine-parseable message so the Java layer can
    // either auto-promote n_ctx (bounded by the memory budget) or show the user a clear error. This
    // check is version-independent: it compares only our own n_prompt_tokens against our own g_n_ctx,
    // so it survives llama.cpp/mtmd internal changes. MIN_GEN_HEADROOM leaves room for a few output
    // tokens (the prompt merely fitting is not enough — there must be slots left to generate into).
    {
        const size_t MIN_GEN_HEADROOM = 16;
        if (n_prompt_tokens + MIN_GEN_HEADROOM > static_cast<size_t>(g_n_ctx)) {
            std::ostringstream es;
            es << "CTX_TOO_SMALL need=" << (n_prompt_tokens + MIN_GEN_HEADROOM)
               << " have=" << g_n_ctx
               << " (multimodal prompt exceeds context; increase Context Size (n_ctx) or use a smaller image)";
            error_message = es.str();
            log_to_file(std::string("prefill_multimodal: ") + error_message, GGML_LOG_LEVEL_ERROR);
            cleanup();
            return false;
        }
    }
    {
        // [mm-diag] Confirm the media marker(s) matched the decoded media and that the
        // multimodal chunks actually carry non-text (image/audio) tokens. If markers != media
        // or there are no media chunks, mtmd would have embedded the media as plain text and
        // the model will say it "cannot see/hear" the input.
        const std::string mm_marker = "<__media__>";
        size_t marker_count = 0, mpos = 0;
        while ((mpos = prompt.find(mm_marker, mpos)) != std::string::npos) {
            marker_count++;
            mpos += mm_marker.size();
        }
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        size_t n_media_chunks = 0;
        for (size_t ci = 0; ci < n_chunks; ci++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, ci);
            if (chunk && mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
                n_media_chunks++;
            }
        }
        std::ostringstream ss;
        ss << "[mm-diag] prefill_multimodal: media=" << media_files.size()
           << " markers=" << marker_count
           << " chunks=" << n_chunks
           << " media_chunks=" << n_media_chunks
           << " n_prompt_tokens=" << n_prompt_tokens;
        log_to_file(ss.str(), GGML_LOG_LEVEL_INFO);
        if (marker_count != media_files.size()) {
            log_to_file("[mm-diag] WARNING: media marker count != media file count", GGML_LOG_LEVEL_ERROR);
        }
        if (n_media_chunks == 0 && !media_files.empty()) {
            log_to_file("[mm-diag] WARNING: no media chunks produced — media embedded as text; model will not perceive it", GGML_LOG_LEVEL_ERROR);
        }
    }
    const int32_t eval_rc = mtmd_helper_eval_chunks(
            g_mtmd,
            g_ctx,
            chunks,
            0,
            0,
            g_n_batch,
            true,
            &n_past
    );
    if (eval_rc != 0) {
        error_message = "decode failed (multimodal prompt)";
        cleanup();
        return false;
    }
    {
        std::ostringstream ss;
        ss << "[mm-diag] prefill_multimodal eval ok: n_past=" << n_past;
        log_to_file(ss.str(), GGML_LOG_LEVEL_INFO);
    }

    cleanup();
    return true;
}

// [mm-diag] Warn when the built prompt still carries media markers but no media reached native
// (media dropped before the JNI call). The model then sees "<__media__>" as literal text and
// reports it cannot see/hear the input — logged so an on-device repro pinpoints where it broke.
static void log_media_marker_diag(const std::string & prompt, bool has_media, size_t media_count, const char * where) {
    const std::string mm_marker = "<__media__>";
    size_t marker_count = 0, mpos = 0;
    while ((mpos = prompt.find(mm_marker, mpos)) != std::string::npos) {
        marker_count++;
        mpos += mm_marker.size();
    }
    std::ostringstream ss;
    ss << "[mm-diag] " << where << ": has_media=" << (has_media ? 1 : 0)
       << " media_count=" << media_count << " markers=" << marker_count;
    log_to_file(ss.str(), GGML_LOG_LEVEL_INFO);
    if (marker_count > 0 && !has_media) {
        log_to_file(std::string("[mm-diag] WARNING: ") + where
                    + ": prompt has media markers but NO media reached native — media dropped upstream; model sees it as text.",
                    GGML_LOG_LEVEL_ERROR);
    }
}

static jstring generate_locked(
        JNIEnv * env,
        const std::string & prompt,
        const std::vector<std::vector<uint8_t>> * media_files) {
    if (!g_ctx || !g_model) {
        log_to_file("generate: not initialized", GGML_LOG_LEVEL_ERROR);
        notify_token_error("not initialized");
        return new_java_string_utf8(env, "not initialized");
    }

    g_cancel_generation.store(false);

    {
        std::ostringstream ss;
        ss << "generate: prompt_len=" << prompt.size()
           << " media_count=" << (media_files ? media_files->size() : 0);
        log_to_file(ss.str());
    }
    if (is_max_debug_mode()) {
        log_to_file(std::string("generate: prompt=\n") + prompt, GGML_LOG_LEVEL_DEBUG);
    }

    const int max_tokens = (g_n_predict > 0) ? g_n_predict : (g_n_ctx - 32);
    std::string context_error;
    if (!reset_generation_context_locked("generate", context_error)) {
        log_to_file(std::string("generate: ") + context_error, GGML_LOG_LEVEL_ERROR);
        notify_token_error(context_error.c_str());
        return new_java_string_utf8(env, context_error);
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    size_t n_prompt_tokens = 0;
    llama_pos n_past = 0;
    std::string prefill_error;
    reusable_token_batch generation_batch(1);
    if (!generation_batch.valid()) {
        const char * errmsg = "failed to allocate generation batch";
        log_to_file(std::string("generate: ") + errmsg, GGML_LOG_LEVEL_ERROR);
        notify_token_error(errmsg);
        return new_java_string_utf8(env, errmsg);
    }

    const bool has_media = media_files != nullptr && !media_files->empty();
    log_media_marker_diag(prompt, has_media, media_files ? media_files->size() : 0, "generate");
    const bool prefill_ok = has_media
            ? prefill_multimodal_prompt_locked(prompt, *media_files, n_prompt_tokens, n_past, prefill_error)
            : prefill_text_prompt_locked(vocab, prompt, n_prompt_tokens, n_past, prefill_error);

    if (!prefill_ok) {
        log_to_file(std::string("generate: prompt prefill failed: ") + prefill_error, GGML_LOG_LEVEL_ERROR);
        // A CTX_TOO_SMALL shortfall is a retryable/handleable pre-generation condition: the Java
        // layer may auto-promote n_ctx and retry, or surface a clean final message via the return
        // value. Do NOT push it to the token stream — a mid-stream error frame followed by a silent
        // retry corrupts the WebUI SSE state machine (browser freeze). The error is still returned
        // as the function result so the caller can act on it. Other prefill errors are terminal and
        // still notify the stream so a streaming client sees them.
        if (prefill_error.rfind("CTX_TOO_SMALL", 0) != 0) {
            notify_token_error(prefill_error.c_str());
        }
        return new_java_string_utf8(env, prefill_error);
    }

    if (n_past >= g_n_ctx) {
        const char * errmsg = "prompt exceeds context";
        log_to_file(std::string("generate: ") + errmsg, GGML_LOG_LEVEL_WARN);
        notify_token_error(errmsg);
        return new_java_string_utf8(env, errmsg);
    }

    const llama_pos generation_base_pos = n_past;
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // Structured output: when a grammar is active, use common_sampler — the proven path that
    // correctly applies GBNF / JSON-schema grammars (same as generateOpenAiChatCompletion). A raw
    // llama_sampler_init_grammar in the manual chain aborts in this fork, so it must not be used.
    // Otherwise fall back to the manual sampler chain (unchanged for normal generation).
    common_sampler_ptr gsmpl;
    llama_sampler * smpl = nullptr;
    if (!g_grammar.empty()) {
        // A malformed JSON schema / GBNF makes common_sampler_init throw (json_schema_to_grammar
        // raises std::runtime_error, and the prefill path in common_sampler_init re-throws).
        // generate_locked has no outer try, so an uncaught throw would cross the JNI boundary and
        // SIGABRT the whole app. Catch it here, and fail loudly rather than silently producing
        // unconstrained output the caller explicitly asked to be structured.
        try {
            common_params_sampling sp = build_chat_sampling_params_locked(common_chat_params{});
            sp.grammar = g_grammar;
            sp.grammar_lazy = false;
            gsmpl.reset(common_sampler_init(g_model, sp));
        } catch (const std::exception & e) {
            const std::string errmsg = std::string("invalid structured-output constraint: ") + e.what();
            log_to_file(std::string("generate: ") + errmsg, GGML_LOG_LEVEL_ERROR);
            notify_token_error(errmsg.c_str());
            return new_java_string_utf8(env, errmsg);
        }
        if (!gsmpl) {
            const char * errmsg = "invalid structured-output constraint (grammar/schema could not be compiled)";
            log_to_file(std::string("generate: ") + errmsg, GGML_LOG_LEVEL_ERROR);
            notify_token_error(errmsg);
            return new_java_string_utf8(env, errmsg);
        }
    }

    if (!gsmpl) {
    auto sparams = llama_sampler_chain_default_params();
    smpl = llama_sampler_chain_init(sparams);

    if (g_penalty_last_n > 0 && (g_penalty_repeat != 1.0f || g_penalty_freq != 0.0f || g_penalty_present != 0.0f)) {
        // b10621/v0.3.0 re-added n_vocab as the leading argument.
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
                llama_vocab_n_tokens(vocab),
                g_penalty_last_n, g_penalty_repeat, g_penalty_freq, g_penalty_present));
    }

    if (g_dry_multiplier > 0.0f) {
        std::vector<std::string> breaker_strings;
        std::vector<const char *> breaker_ptrs;
        std::string temp = g_dry_sequence_breakers;
        size_t pos = 0;
        while ((pos = temp.find(',')) != std::string::npos) {
            std::string token = temp.substr(0, pos);
            if (!token.empty()) {
                breaker_strings.push_back(process_escape_sequences(token));
            }
            temp.erase(0, pos + 1);
        }
        if (!temp.empty()) {
            breaker_strings.push_back(process_escape_sequences(temp));
        }
        for (const auto & s : breaker_strings) {
            breaker_ptrs.push_back(s.c_str());
        }
        if (!breaker_ptrs.empty()) {
            // b10621/v0.3.0 dropped the n_ctx_train argument from the dry sampler.
            llama_sampler_chain_add(smpl, llama_sampler_init_dry(
                    vocab, g_dry_multiplier, g_dry_base,
                    g_dry_allowed_length, g_dry_penalty_last_n,
                    breaker_ptrs.data(), breaker_ptrs.size()));
        }
    }

    if (g_top_n_sigma > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_n_sigma(g_top_n_sigma));
    }

    // Build the ordered sampler list from g_sampler_order (if set) or the default order.
    // Penalties, DRY, and top_n_sigma are always applied first (above); the list here
    // covers the probabilistic samplers that the WebUI "Samplers" setting can reorder.
    std::vector<std::string> sampler_names;
    if (!g_sampler_order.empty()) {
        std::stringstream ss(g_sampler_order);
        std::string token;
        while (std::getline(ss, token, ';')) {
            if (!token.empty()) sampler_names.push_back(token);
        }
    } else {
        sampler_names = {"top_k", "typ_p", "top_p", "min_p", "temperature"};
    }
    // Reset order after reading so that subsequent calls without an explicit order
    // fall back to the default (avoids stale order leaking into direct-input calls).
    g_sampler_order = "";

    for (const auto& name : sampler_names) {
        if (name == "top_k") {
            if (g_top_k > 0)
                llama_sampler_chain_add(smpl, llama_sampler_init_top_k(g_top_k));
        } else if (name == "typ_p" || name == "typical_p") {
            if (g_typical_p < 1.0f)
                llama_sampler_chain_add(smpl, llama_sampler_init_typical(g_typical_p, 1));
        } else if (name == "top_p") {
            if (g_top_p < 1.0f)
                llama_sampler_chain_add(smpl, llama_sampler_init_top_p(g_top_p, 1));
        } else if (name == "min_p") {
            if (g_min_p > 0.0f)
                llama_sampler_chain_add(smpl, llama_sampler_init_min_p(g_min_p, 1));
        } else if (name == "xtc") {
            if (g_xtc_probability > 0.0f)
                llama_sampler_chain_add(smpl, llama_sampler_init_xtc(
                        g_xtc_probability, g_xtc_threshold, 1, LLAMA_DEFAULT_SEED));
        } else if (name == "temperature" || name == "temp") {
            if (g_dynatemp_range > 0.0f)
                llama_sampler_chain_add(smpl, llama_sampler_init_temp_ext(
                        g_temp, g_dynatemp_range, g_dynatemp_exponent));
            else
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(g_temp));
        }
    }

    // Final sampler: mirostat or dist (always last)
    if (g_mirostat == 1) {
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
                n_vocab, LLAMA_DEFAULT_SEED, g_mirostat_tau, g_mirostat_eta, 100));
    } else if (g_mirostat == 2) {
        llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
                LLAMA_DEFAULT_SEED, g_mirostat_tau, g_mirostat_eta));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }
    } // end manual sampler chain (used when no grammar is active)

    // MTP speculative decoding (draft-mtp): active for text generation when g_spec is set.
    // sample_and_accept_n needs a common_sampler, so if none was built above (no grammar),
    // build one from the same g_* sampling params. When inactive the loop is byte-identical.
    const bool spec_active = (g_spec != nullptr) && !has_media;
    if (spec_active && !gsmpl) {
        try {
            // common_sampler_init takes params by non-const ref, so bind to an lvalue.
            common_params_sampling sp = build_chat_sampling_params_locked(common_chat_params{});
            gsmpl.reset(common_sampler_init(g_model, sp));
        } catch (...) { gsmpl.reset(); }
    }
    const bool use_spec = spec_active && (bool) gsmpl;
    std::vector<llama_token> spec_buf;   // accepted tokens from the current step
    size_t      spec_pos = 0;
    llama_token spec_id_last = 0;
    llama_pos   spec_n_past = generation_base_pos;
    common_prompt_checkpoint spec_ckpt;
    bool spec_bootstrap = true;
    bool spec_failed = false;
    if (use_spec) {
        g_spec_prompt.clear();
        common_speculative_begin(g_spec, 0, g_spec_prompt);
        g_spec_stat_steps = g_spec_stat_out = g_spec_stat_eval = 0;
        log_to_file("generate: MTP speculative decoding active");
    }

    std::string output;
    output.reserve(max_tokens * 4);
    std::vector<llama_token> out_tokens;
    out_tokens.reserve(max_tokens);
    std::string prev_text;
    bool logged_first_sampling_step = false;
    bool logged_first_decode_step = false;
    const int64_t gen_wall_t0_us = ggml_time_us();  // wall-clock start of generation

    for (int i = 0; i < max_tokens; ++i) {
        if (g_cancel_generation.load()) {
            break;
        }

        if (!logged_first_sampling_step) {
            log_to_file("generate: sampling first token");
            logged_first_sampling_step = true;
        }

        llama_token id;
        if (use_spec) {
            if (spec_bootstrap) {
                // First output token: sample from the prefill logits. Its id seeds the
                // first speculative step (which decodes it), so it is NOT decoded here.
                id = common_sampler_sample(gsmpl.get(), g_ctx, -1);
                common_sampler_accept(gsmpl.get(), id, true);
                spec_id_last = id;
                spec_bootstrap = false;
            } else {
                if (spec_pos >= spec_buf.size()) {
                    spec_buf.clear();
                    spec_pos = 0;
                    if (!spec_decode_step_locked(gsmpl, spec_id_last, spec_n_past, spec_ckpt, spec_buf)
                            || spec_buf.empty()) {
                        spec_failed = true;
                        break;
                    }
                }
                id = spec_buf[spec_pos++];
            }
        } else {
            id = gsmpl ? common_sampler_sample(gsmpl.get(), g_ctx, -1)
                       : llama_sampler_sample(smpl, g_ctx, -1);
            if (gsmpl) {
                common_sampler_accept(gsmpl.get(), id, true);
            } else {
                llama_sampler_accept(smpl, id);
            }
        }
        if (i == 0) {
            log_to_file("generate: first token sampled id=" + std::to_string(id));
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }

        out_tokens.push_back(id);
        if (generation_base_pos + static_cast<llama_pos>(out_tokens.size()) >= g_n_ctx - 32) {
            log_to_file("generate: reached ctx safety limit, stopping early");
            break;
        }

        std::string full;
        int n_chars = detokenize_with_resize(vocab, out_tokens, full);
        if (n_chars > 0) {
            size_t safe_len = validate_utf8(full);
            if (safe_len < full.size()) {
                full.resize(safe_len);
            }

            bool stop_hit = false;
            size_t stop_pos = std::string::npos;
            std::string stop_seq;
            if (find_stop_sequence(full, &stop_pos, &stop_seq)) {
                if (stop_pos < full.size()) {
                    full.resize(stop_pos);
                }
                stop_hit = true;
            }

            std::string delta;
            if (full.size() > prev_text.size()) {
                delta = full.substr(prev_text.size());
                output += delta;
            }
            if (stop_hit) {
                output = full;
            }

            if (!delta.empty()) {
                notify_token_delta(delta);
            }

            prev_text = full;
            if (stop_hit) {
                break;
            }
        }

        // Non-speculative path decodes the sampled token here. In the MTP path the token
        // was already committed to the KV inside spec_decode_step_locked, so skip this.
        if (!use_spec) {
            generation_batch.batch.n_tokens = 1;
            generation_batch.set_token(0, id, generation_base_pos + i, true);

            if (!logged_first_decode_step) {
                log_to_file("generate: decoding first sampled token");
                logged_first_decode_step = true;
            }
            const int rc_step = llama_decode(g_ctx, generation_batch.batch);
            if (rc_step != 0) {
                const char * errmsg = "decode failed (generation)";
                log_to_file(std::string("generate: ") + errmsg, GGML_LOG_LEVEL_ERROR);
                notify_token_error(errmsg);
                if (smpl) llama_sampler_free(smpl);
                return new_java_string_utf8(env, errmsg);
            }
            if (i == 0) {
                log_to_file("generate: first sampled token decode complete");
            }
        }
    }

    if (spec_failed) {
        log_to_file("generate: MTP speculative step failed; ending generation early", GGML_LOG_LEVEL_WARN);
    }
    if (use_spec && g_spec_stat_steps > 0) {
        const double eval_per_out = (double) g_spec_stat_eval / (double) std::max(1L, g_spec_stat_out);
        std::ostringstream ss;
        ss << "generate: MTP stats steps=" << g_spec_stat_steps
           << " out_tokens=" << g_spec_stat_out
           << " decoded_positions=" << g_spec_stat_eval
           << " (eval/out=" << eval_per_out << "x; >1 means speculation decoded more"
           << " positions than it produced -> net slower on a compute-bound CPU)";
        log_to_file(ss.str());
    }

    log_generation_perf("generate");
    if (use_spec) {
        // llama_perf attributes MTP's multi-token draft batches to prompt-eval (n_p_eval),
        // so its gen tok/s reads ~0. Report/store the real wall-clock generation speed.
        const double gen_s = (ggml_time_us() - gen_wall_t0_us) / 1.0e6;
        const int n_gen = (int) out_tokens.size();
        const double gen_tps = (gen_s > 0.0) ? (n_gen / gen_s) : 0.0;
        g_last_gen_tps.store(gen_tps, std::memory_order_relaxed);
        g_last_n_eval.store(n_gen, std::memory_order_relaxed);
        std::ostringstream ss;
        ss << "generate: speed (MTP wall-clock): gen " << std::fixed << std::setprecision(2)
           << gen_tps << " tok/s (" << n_gen << " tok / " << gen_s << " s)";
        log_to_file(ss.str());
    }

    const bool was_cancelled = g_cancel_generation.load();
    if (!was_cancelled) {
        notify_token_complete();
    } else {
        log_to_file("generate: completed with cancellation", GGML_LOG_LEVEL_WARN);
    }

    if (smpl) llama_sampler_free(smpl);

    size_t safe_len = validate_utf8(output);
    if (safe_len < output.size()) {
        output.resize(safe_len);
    }

    if (is_max_debug_mode()) {
        log_to_file(std::string("generate: output=\n") + output, GGML_LOG_LEVEL_DEBUG);
    }

    return new_java_string_utf8(env, output);
}

static jstring generate_openai_chat_completion_locked(
        JNIEnv * env,
        const std::string & messages_json_text,
        const std::string & tools_json_text,
        const std::string & chat_template_override,
        const std::string & tool_choice_json_text,
        bool parallel_tool_calls,
        bool enable_thinking,
        const std::vector<std::vector<uint8_t>> * media_files) {
    auto build_error = [&](const std::string & message) -> jstring {
        log_to_file(std::string("generateOpenAiChatCompletion: ") + message, GGML_LOG_LEVEL_ERROR);
        json error_json = {
                {"error", message},
        };
        return new_java_string_utf8(env, error_json.dump());
    };

    if (!g_ctx || !g_model) {
        return build_error("not initialized");
    }

    g_cancel_generation.store(false);

    try {
        // b10621/v0.3.0 replaced nlohmann::json with the common_json abstraction in the
        // common_chat_* / json_schema_to_grammar APIs; parse straight into common_json.
        // tool_choice still needs nlohmann here for its object-shape inspection below (the
        // common API itself takes a plain std::string).
        common_json messages_json = common_json::parse(messages_json_text.empty() ? "[]" : messages_json_text);
        common_json tools_json = common_json::parse(tools_json_text.empty() ? "[]" : tools_json_text);
        json tool_choice_json = tool_choice_json_text.empty() ? json() : json::parse(tool_choice_json_text);

        auto messages = common_chat_msgs_parse_oaicompat(messages_json);
        auto tools = common_chat_tools_parse_oaicompat(tools_json);
        common_chat_tool_choice tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

        if (!tool_choice_json.is_null()) {
            if (tool_choice_json.is_string()) {
                tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice_json.get<std::string>());
            } else if (tool_choice_json.is_object()) {
                if (tool_choice_json.value("type", std::string()) != "function") {
                    throw std::runtime_error("unsupported tool_choice object type");
                }
                const auto & function = tool_choice_json.at("function");
                const std::string forced_tool_name = function.at("name").get<std::string>();
                std::vector<common_chat_tool> filtered_tools;
                for (const auto & tool : tools) {
                    if (tool.name == forced_tool_name) {
                        filtered_tools.push_back(tool);
                    }
                }
                if (filtered_tools.empty()) {
                    throw std::runtime_error("tool_choice function not found in tools");
                }
                tools = std::move(filtered_tools);
                tool_choice = COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            } else {
                throw std::runtime_error("tool_choice must be a string or object");
            }
        }

        common_chat_templates_inputs inputs;
        inputs.messages = std::move(messages);
        inputs.tools = std::move(tools);
        inputs.tool_choice = tool_choice;
        inputs.parallel_tool_calls = parallel_tool_calls;
        inputs.enable_thinking = enable_thinking;
        if (enable_thinking) {
            inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        }
        inputs.add_generation_prompt = true;
        inputs.use_jinja = true;

        auto chat_templates = common_chat_templates_init(g_model, chat_template_override);
        auto chat_params = common_chat_templates_apply(chat_templates.get(), inputs);

        common_chat_parser_params parser_params(chat_params);
        parser_params.parse_tool_calls = true;
        if (!chat_params.parser.empty()) {
            parser_params.parser.load(chat_params.parser);
        }

        const std::string & prompt = chat_params.prompt;
        const auto stop_sequences = build_chat_stop_sequences(chat_params);

        if (is_max_debug_mode()) {
            log_to_file(std::string("generateOpenAiChatCompletion: prompt=\n") + prompt, GGML_LOG_LEVEL_DEBUG);
        }

        const llama_vocab * vocab = llama_model_get_vocab(g_model);
        size_t n_prompt_tokens = 0;
        llama_pos n_past = 0;
        std::string prefill_error;
        reusable_token_batch generation_batch(1);
        if (!generation_batch.valid()) {
            return build_error("failed to allocate generation batch");
        }

        const bool has_media = media_files != nullptr && !media_files->empty();
        log_media_marker_diag(prompt, has_media, media_files ? media_files->size() : 0, "generateOpenAiChatCompletion");
        // Reuse the KV cache across turns (MCP) for the text path on CPU/GPU; otherwise full reset.
        const bool use_kv_cache = !has_media && kv_prefix_cache_allowed_locked();
        bool prefill_ok;
        if (use_kv_cache) {
            prefill_ok = prefill_text_prompt_cached_locked(vocab, prompt, n_prompt_tokens, n_past, prefill_error);
        } else {
            std::string context_error;
            if (!reset_generation_context_locked("generateOpenAiChatCompletion", context_error)) {
                return build_error(context_error);
            }
            prefill_ok = has_media
                    ? prefill_multimodal_prompt_locked(prompt, *media_files, n_prompt_tokens, n_past, prefill_error)
                    : prefill_text_prompt_locked(vocab, prompt, n_prompt_tokens, n_past, prefill_error);
        }

        if (!prefill_ok) {
            return build_error(prefill_error);
        }

        if (n_past >= g_n_ctx) {
            return build_error("prompt exceeds context");
        }

        common_params_sampling sampling_params = build_chat_sampling_params_locked(chat_params);
        common_sampler_ptr sampler(common_sampler_init(g_model, sampling_params));
        if (!sampler) {
            return build_error("failed to initialize sampler");
        }

        const llama_pos generation_base_pos = n_past;
        const int max_tokens = (g_n_predict > 0) ? g_n_predict : (g_n_ctx - (int)n_past - 32);

        std::string output;
        output.reserve(std::max(max_tokens, 1) * 4);
        std::vector<llama_token> out_tokens;
        out_tokens.reserve(std::max(max_tokens, 1));
        std::string prev_text;

        for (int i = 0; i < max_tokens; ++i) {
            if (g_cancel_generation.load()) {
                break;
            }

            const llama_token id = common_sampler_sample(sampler.get(), g_ctx, -1);
            common_sampler_accept(sampler.get(), id, true);
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

            out_tokens.push_back(id);
            if (generation_base_pos + static_cast<llama_pos>(out_tokens.size()) >= g_n_ctx - 32) {
                log_to_file("generateOpenAiChatCompletion: reached ctx safety limit, stopping early");
                break;
            }

            std::string full;
            int n_chars = detokenize_with_resize(vocab, out_tokens, full);
            if (n_chars > 0) {
                size_t safe_len = validate_utf8(full);
                if (safe_len < full.size()) {
                    full.resize(safe_len);
                }

                size_t stop_pos = std::string::npos;
                std::string stop_seq;
                if (find_stop_sequence_in_list(stop_sequences, full, &stop_pos, &stop_seq)) {
                    if (stop_pos < full.size()) {
                        full.resize(stop_pos);
                    }
                    output = full;
                    prev_text = full;
                    break;
                }

                if (full.size() > prev_text.size()) {
                    output += full.substr(prev_text.size());
                }
                prev_text = full;
            }

            generation_batch.batch.n_tokens = 1;
            generation_batch.set_token(0, id, generation_base_pos + i, true);

            const int rc_step = llama_decode(g_ctx, generation_batch.batch);
            if (rc_step != 0) {
                g_kv_cached_tokens.clear();
                return build_error("decode failed (generation)");
            }
            if (use_kv_cache) {
                g_kv_cached_tokens.push_back(id);   // keep the prefix cache in sync with the KV
            }
        }

        log_generation_perf("generateOpenAiChatCompletion");

        size_t safe_len = validate_utf8(output);
        if (safe_len < output.size()) {
            output.resize(safe_len);
        }

        common_chat_msg parsed_msg;
        try {
            parsed_msg = common_chat_parse(output, false, parser_params);
        } catch (const std::exception & e) {
            log_to_file(std::string("generateOpenAiChatCompletion: parser fallback: ") + e.what(), GGML_LOG_LEVEL_WARN);
            parsed_msg.content = output;
        }
        if (parsed_msg.empty() && !output.empty()) {
            parsed_msg.content = output;
        }

        json result = {
                {"content", parsed_msg.content},
                {"finish_reason", parsed_msg.tool_calls.empty() ? "stop" : "tool_calls"},
        };
        if (!parsed_msg.reasoning_content.empty()) {
            result["reasoning_content"] = parsed_msg.reasoning_content;
        }
        if (!parsed_msg.tool_calls.empty()) {
            auto msg_json = parsed_msg.to_json_oaicompat(false);
            result["tool_calls"] = msg_json["tool_calls"];
        }

        if (is_max_debug_mode()) {
            log_to_file(std::string("generateOpenAiChatCompletion: result=\n") + result.dump(2), GGML_LOG_LEVEL_DEBUG);
        }

        return new_java_string_utf8(env, result.dump());
    } catch (const std::exception & e) {
        return build_error(e.what());
    }
}

// ---------------- JNI: generate ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_generate(
        JNIEnv *env, jobject,
        jstring jPrompt
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return generate_locked(env, jstring_to_std(env, jPrompt), nullptr);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_generateWithMedia(
        JNIEnv *env, jobject,
        jstring jPrompt,
        jobjectArray jMediaFiles
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    std::vector<std::vector<uint8_t>> media_files;
    if (jMediaFiles != nullptr) {
        const jsize count = env->GetArrayLength(jMediaFiles);
        media_files.reserve(static_cast<size_t>(count));
        for (jsize i = 0; i < count; ++i) {
            auto * byte_array = reinterpret_cast<jbyteArray>(env->GetObjectArrayElement(jMediaFiles, i));
            if (!byte_array) {
                media_files.emplace_back();
                continue;
            }
            const jsize len = env->GetArrayLength(byte_array);
            std::vector<uint8_t> bytes(static_cast<size_t>(len));
            if (len > 0) {
                env->GetByteArrayRegion(byte_array, 0, len, reinterpret_cast<jbyte *>(bytes.data()));
            }
            media_files.push_back(std::move(bytes));
            env->DeleteLocalRef(byte_array);
        }
    }

    return generate_locked(env, jstring_to_std(env, jPrompt), &media_files);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_generateOpenAiChatCompletion(
        JNIEnv *env, jobject,
        jstring jMessagesJson,
        jstring jToolsJson,
        jstring jChatTemplateOverride,
        jstring jToolChoiceJson,
        jboolean jParallelToolCalls,
        jboolean jEnableThinking,
        jobjectArray jMediaFiles
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    std::vector<std::vector<uint8_t>> media_files;
    if (jMediaFiles != nullptr) {
        const jsize count = env->GetArrayLength(jMediaFiles);
        media_files.reserve(static_cast<size_t>(count));
        for (jsize i = 0; i < count; ++i) {
            auto * byte_array = reinterpret_cast<jbyteArray>(env->GetObjectArrayElement(jMediaFiles, i));
            if (!byte_array) {
                media_files.emplace_back();
                continue;
            }
            const jsize len = env->GetArrayLength(byte_array);
            std::vector<uint8_t> bytes(static_cast<size_t>(len));
            if (len > 0) {
                env->GetByteArrayRegion(byte_array, 0, len, reinterpret_cast<jbyte *>(bytes.data()));
            }
            media_files.push_back(std::move(bytes));
            env->DeleteLocalRef(byte_array);
        }
    }

    return generate_openai_chat_completion_locked(
            env,
            jstring_to_std(env, jMessagesJson),
            jstring_to_std(env, jToolsJson),
            jstring_to_std(env, jChatTemplateOverride),
            jstring_to_std(env, jToolChoiceJson),
            jParallelToolCalls == JNI_TRUE,
            jEnableThinking == JNI_TRUE,
            &media_files);
}

// ---------------- JNI: free ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_free(
        JNIEnv *env, jobject /*thiz*/
) {
    log_to_file("Java_com_micklab_llama_LlamaNative_free called");
    llama_jni_free();
}

// ---------------- JNI: getChatTemplate ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_getChatTemplate(
        JNIEnv *env, jobject /*thiz*/
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (!g_model) {
        log_to_file("getChatTemplate: model not loaded", GGML_LOG_LEVEL_WARN);
        return new_java_string_utf8(env, "");
    }
    
    // Try to get chat template from GGUF metadata
    const char* chat_template = llama_model_chat_template(g_model, nullptr);
    
    if (chat_template && strlen(chat_template) > 0) {
        log_to_file(std::string("getChatTemplate: found template, len=") + std::to_string(strlen(chat_template)));
        return new_java_string_utf8(env, chat_template);
    }
    
    log_to_file("getChatTemplate: no chat template in model metadata");
    return new_java_string_utf8(env, "");
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_micklab_llama_LlamaNative_supportsVision(
        JNIEnv *, jobject /*thiz*/
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_supports_vision ? JNI_TRUE : JNI_FALSE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_micklab_llama_LlamaNative_supportsAudio(
        JNIEnv *, jobject /*thiz*/
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_supports_audio ? JNI_TRUE : JNI_FALSE;
}

// 直近の生成のデコード速度 (tok/s) を返す。UI のステータス表示用。
extern "C"
JNIEXPORT jdouble JNICALL
Java_com_micklab_llama_LlamaNative_getLastGenerationSpeed(
        JNIEnv *, jobject /*thiz*/
) {
    return (jdouble) g_last_gen_tps.load(std::memory_order_relaxed);
}

// 直近の生成の入力トークン数を返す。
extern "C"
JNIEXPORT jint JNICALL
Java_com_micklab_llama_LlamaNative_getLastNPromptTokens(
        JNIEnv *, jobject /*thiz*/
) {
    return (jint) g_last_n_prompt.load(std::memory_order_relaxed);
}

// 直近の生成の出力トークン数を返す。
extern "C"
JNIEXPORT jint JNICALL
Java_com_micklab_llama_LlamaNative_getLastNEvalTokens(
        JNIEnv *, jobject /*thiz*/
) {
    return (jint) g_last_n_eval.load(std::memory_order_relaxed);
}

// KV キャッシュ 1 セル (= 1 トークン) あたりのバイト数を返す (0 = モデル未ロード)。
// Q2 の自動昇格 cap 計算に使う: あるモデル・KV量子化型で n_ct=N にした際の KV サイズは
// getKvBytesPerCell() * N。実測 (Qwen3VL-2B, f16) で 112KiB/cell に一致する。
// KV(token) = n_layer * n_embd_gqa * (sizeof(type_k) + sizeof(type_v)) ,
//   n_embd_gqa = (n_embd / n_head) * n_head_kv   (n_embd_head_k ≈ n_embd_head_v ≈ n_embd/n_head)
extern "C"
JNIEXPORT jlong JNICALL
Java_com_micklab_llama_LlamaNative_getKvBytesPerCell(
        JNIEnv *, jobject /*thiz*/
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model) {
        return 0;
    }
    const int n_layer   = llama_model_n_layer(g_model);
    const int n_head    = llama_model_n_head(g_model);
    const int n_head_kv = llama_model_n_head_kv(g_model);
    const int n_embd    = llama_model_n_embd(g_model);
    if (n_layer <= 0 || n_head <= 0 || n_head_kv <= 0 || n_embd <= 0) {
        return 0;
    }
    const long n_embd_head = (long) n_embd / n_head;
    const long n_embd_gqa  = n_embd_head * n_head_kv;
    // Bytes per element for the configured KV cache quant types (F16=2, Q8_0≈1.06, …).
    auto bytes_per_elem = [](int type_id) -> double {
        const ggml_type t = (ggml_type) type_id;
        const int blk = ggml_blck_size(t);
        if (blk <= 0) return 2.0;  // fallback to F16 size
        return (double) ggml_type_size(t) / (double) blk;
    };
    const double k_be = bytes_per_elem(g_kv_type_k);
    const double v_be = bytes_per_elem(g_kv_type_v);
    const double per_cell = (double) n_layer * (double) n_embd_gqa * (k_be + v_be);
    return (jlong) (per_cell + 0.5);
}

// 直近の生成の合計時間 (ms) を返す。プロンプト処理 + デコード。
extern "C"
JNIEXPORT jdouble JNICALL
Java_com_micklab_llama_LlamaNative_getLastTotalTimeMs(
        JNIEnv *, jobject /*thiz*/
) {
    return (jdouble) g_last_t_total_ms.load(std::memory_order_relaxed);
}

// 直近の生成のプロンプト処理 (prefill) 時間 (ms) を返す。
extern "C"
JNIEXPORT jdouble JNICALL
Java_com_micklab_llama_LlamaNative_getLastPromptEvalTimeMs(
        JNIEnv *, jobject /*thiz*/
) {
    return (jdouble) g_last_t_p_eval_ms.load(std::memory_order_relaxed);
}

// 直近のモデルロード時間 (ms) を返す。
extern "C"
JNIEXPORT jdouble JNICALL
Java_com_micklab_llama_LlamaNative_getLastLoadTimeMs(
        JNIEnv *, jobject /*thiz*/
) {
    return (jdouble) g_last_t_load_ms.load(std::memory_order_relaxed);
}

// 実効的な計算バックエンド名 ("CPU"/"GPU"/"NPU/HTP"/"GPU+NPU"/"CPU (fallback)") を返す。
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_getActiveBackend(
        JNIEnv * env, jobject /*thiz*/
) {
    // No g_mutex: lock-free read so the UI poller never blocks on an in-flight generation.
    return env->NewStringUTF(g_active_backend.load(std::memory_order_relaxed));
}

// ---------------- JNI: setBackendConfig ----------------
// backendType: 0=CPU  1=GPU  2=NPU/HTP  3=GPU+NPU
// npuEnabled : false = NPU を無効化 (quick toggle; backendType は保持)
// nativeLibDir: context.getApplicationInfo().nativeLibraryDir
//               HTP skel .so の ADSP_LIBRARY_PATH として使用
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setBackendConfig(
        JNIEnv * env, jobject,
        jint backendType, jboolean npuEnabled, jstring jNativeLibDir) {
    std::lock_guard<std::mutex> lock(g_mutex);

    const int  new_type    = static_cast<int>(backendType);
    const bool new_npu     = static_cast<bool>(npuEnabled);
    const std::string new_dir = jstring_to_std(env, jNativeLibDir);

    const bool changed = (new_type != g_backend_type)
                      || (new_npu  != g_npu_enabled)
                      || (new_dir  != g_native_lib_dir);

    g_backend_type    = new_type;
    g_npu_enabled     = new_npu;
    g_native_lib_dir  = new_dir;

#if defined(GGML_USE_HEXAGON)
    // Publish ADSP_LIBRARY_PATH as soon as the native lib dir is known. This is the earliest
    // point in the process — well before the first llama_backend_init, where the Hexagon backend
    // is auto-registered and opens its DSP session (which needs this path to load the HTP skel).
    // A first session attempt without the path fails and is cached for the whole process, which
    // is why a missed CPU-mode startup permanently disables NPU. See init_backend_locked.
    if (!g_native_lib_dir.empty()) {
        setenv("ADSP_LIBRARY_PATH", g_native_lib_dir.c_str(), 1);
    }
#endif

    // 設定された (要求) バックエンドを表示用に即反映する。
    // 実効値 (フォールバック含む) は次回モデル構築時に build_model_params が精緻化する。
    // これをしないと、前回 NPU で "CPU (fallback)" になった表示が CPU/GPU に切り替えても固着する。
    {
        const bool want_npu = new_npu && (new_type == 2 || new_type == 3);
        const bool want_gpu = (new_type == 1 || new_type == 3);
        g_active_backend = (want_npu && want_gpu) ? "GPU+NPU"
                         : want_npu ? "NPU/HTP"
                         : want_gpu ? "GPU"
                         : "CPU";
    }

    if (changed) {
        // Bump version so ensure_backend_initialized_locked re-initializes on next model load
        ++g_backend_config_version;
    }

    static const char * const BACKEND_NAMES[] = {"CPU", "GPU", "NPU/HTP", "GPU+NPU"};
    const char * type_name = (new_type >= 0 && new_type < 4) ? BACKEND_NAMES[new_type] : "UNKNOWN";

    std::ostringstream ss;
    ss << "setBackendConfig: type=" << new_type
       << " (" << type_name << ")"
       << " npuEnabled=" << (new_npu ? "true" : "false")
       << " nativeLibDir=" << new_dir
       << " configVersion=" << g_backend_config_version
       << (changed ? " [config changed]" : " [no change]");
    log_to_file(ss.str());
}

// ---------------- JNI: setNPredict ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setNPredict(JNIEnv *, jobject, jint n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_n_predict = (int) n;
    std::ostringstream ss;
    ss << "setNPredict: n_predict=" << g_n_predict;
    log_to_file(ss.str());
}

// ---------------- JNI: setKvCacheType ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_micklab_llama_LlamaNative_setKvCacheType(JNIEnv *, jobject, jint typeK, jint typeV) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_kv_type_k = (int) typeK;
    g_kv_type_v = (int) typeV;
    std::ostringstream ss;
    ss << "setKvCacheType: type_k=" << g_kv_type_k << " type_v=" << g_kv_type_v;
    log_to_file(ss.str());
}

// ---------------- JNI: tokenize ----------------
// Returns JSON: {"tokens":["tok1","tok2",...],"count":N,"ids":[1,2,...]}
// or {"error":"..."} if no model is loaded.
extern "C"
JNIEXPORT jstring JNICALL
Java_com_micklab_llama_LlamaNative_tokenize(JNIEnv *env, jobject, jstring jText) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        return new_java_string_utf8(env, "{\"error\":\"no model loaded\"}");
    }
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) {
        return new_java_string_utf8(env, "{\"error\":\"vocab unavailable\"}");
    }
    std::string text = jText ? jstring_to_std(env, jText) : "";

    // Tokenize with add_special=true, parse_special=true to handle BOS/EOS.
    std::vector<llama_token> ids(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           ids.data(), (int32_t)ids.size(), true, true);
    if (n < 0) {
        // Buffer too small — retry with exact size
        ids.resize(-n + 4);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                           ids.data(), (int32_t)ids.size(), true, true);
    }
    if (n < 0) {
        return new_java_string_utf8(env, "{\"error\":\"tokenize failed\"}");
    }
    ids.resize(n);

    // Build JSON manually for speed
    std::ostringstream json;
    json << "{\"tokens\":[";
    for (int i = 0; i < n; ++i) {
        if (i > 0) json << ",";
        char buf[256];
        int len = llama_token_to_piece(vocab, ids[i], buf, (int)sizeof(buf) - 1, 0, true);
        if (len < 0) len = 0;
        buf[len] = '\0';
        // JSON-escape the piece
        json << "\"";
        for (int c = 0; c < len; ++c) {
            unsigned char ch = (unsigned char) buf[c];
            if (ch == '\"') json << "\\\"";
            else if (ch == '\\') json << "\\\\";
            else if (ch == '\n') json << "\\n";
            else if (ch == '\r') json << "\\r";
            else if (ch == '\t') json << "\\t";
            else if (ch < 0x20) { json << "\\u00"; json << "0123456789abcdef"[ch >> 4]; json << "0123456789abcdef"[ch & 0xf]; }
            else json << (char)ch;
        }
        json << "\"";
    }
    json << "],\"count\":" << n << ",\"ids\":[";
    for (int i = 0; i < n; ++i) {
        if (i > 0) json << ",";
        json << ids[i];
    }
    json << "]}";
    return new_java_string_utf8(env, json.str());
}

extern "C"
JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM *, void *) {
    std::lock_guard<std::mutex> lock(g_mutex);

    release_multimodal_locked("JNI_OnUnload");
    release_context_locked("JNI_OnUnload");
    release_model_locked("JNI_OnUnload");
    g_current_mmproj_path.clear();
    g_supports_vision = false;
    g_supports_audio = false;
    g_current_audio_requested = false;
    release_backend_locked("JNI_OnUnload");
}
