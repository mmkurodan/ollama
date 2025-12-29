#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>

#include <android/log.h>
#define LOG_TAG "LLAMA_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include "llama.h"
#include <curl/curl.h>

// ---------------- グローバル ----------------
static std::mutex g_mutex;
static llama_model   *g_model = nullptr;
static llama_context *g_ctx   = nullptr;
static JavaVM *g_jvm = nullptr;

// ログ用
static std::mutex g_log_mutex;
static std::string g_log_path;
static std::ofstream g_log_ofs;

// 設定
static int   g_n_ctx      = 512;
static int   g_n_threads  = 2;
static int   g_n_batch    = 16;
static float g_temp       = 0.7f;
static float g_top_p      = 0.9f;
static int   g_top_k      = 40;

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

static void log_to_file(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_path.empty()) return;
    if (!g_log_ofs.is_open()) {
        g_log_ofs.open(g_log_path, std::ios::app | std::ios::binary);
    }
    if (!g_log_ofs) return;
    g_log_ofs << current_time_str() << " [JNI] " << msg << std::endl;
    // flush immediately so logs appear even if app crashes
    g_log_ofs.flush();
}

// ---------------- 既存ユーティリティ ----------------
static std::string jstring_to_std(JNIEnv *env, jstring jstr) {
    if (!jstr) return "";
    const char *chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars ? chars : "");
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

static void throw_java_exception(JNIEnv *env, const char *msg) {
    jclass exClass = env->FindClass("java/lang/RuntimeException");
    if (exClass) env->ThrowNew(exClass, msg);
}

// ---------------- download() 用 ----------------
static size_t write_data(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::ofstream* ofs = reinterpret_cast<std::ofstream*>(userdata);
    ofs->write(reinterpret_cast<const char*>(ptr), size * nmemb);
    return size * nmemb;
}

struct ProgressData {
    jobject thiz_global;
    jmethodID onProgressMethod;
    int last_percent;
};

// curl transfer progress callback (libcurl >= 7.32.0 uses xferinfo)
// signature: int func(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow)
static int xferinfo(void* p, curl_off_t dltotal, curl_off_t dlnow, curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    ProgressData* pd = reinterpret_cast<ProgressData*>(p);
    if (!pd) return 0;
    if (dltotal <= 0) return 0;

    int percent = (int)((dlnow * 100) / dltotal);
    if (percent == pd->last_percent) return 0;
    pd->last_percent = percent;

    // Obtain JNIEnv for this thread
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
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }

    // ログへ出力（スレッドからでも書き込める）
    {
        std::ostringstream ss;
        ss << "Download progress: " << percent << "%";
        log_to_file(ss.str());
    }

    if (attached) {
        g_jvm->DetachCurrentThread();
    }

    return 0;
}

// ---------------- 解放 ----------------
static void llama_jni_free() {
    std::lock_guard<std::mutex> lock(g_mutex);

    log_to_file("llama_jni_free: freeing resources");

    if (g_ctx) {
        // free context
        llama_free(g_ctx);
        g_ctx = nullptr;
        log_to_file("Context freed");
    }
    if (g_model) {
        // free model
        llama_free_model(g_model);
        g_model = nullptr;
        log_to_file("Model freed");
    }

    // backend cleanup
    llama_backend_free();
    log_to_file("Backend freed");

    // close log file if open
    std::lock_guard<std::mutex> llog(g_log_mutex);
    if (g_log_ofs.is_open()) {
        g_log_ofs << current_time_str() << " [JNI] Log closed" << std::endl;
        g_log_ofs.close();
    }
    g_log_path.clear();
}

// ---------------- JNI: setLogPath ----------------
// call from Java with app-specific external directory path (recommended): e.g. context.getExternalFilesDir(null).getAbsolutePath() + "/ollama.log"
extern "C"
JNIEXPORT void JNICALL
Java_com_example_ollama_LlamaNative_setLogPath(
        JNIEnv *env, jobject /*thiz*/, jstring jLogPath) {

    std::string path = jstring_to_std(env, jLogPath);
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
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
    }
}

// ---------------- JNI: download ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_download(
        JNIEnv* env,
        jobject thiz,
        jstring jurl,
        jstring jpath) {

    const char* url  = env->GetStringUTFChars(jurl,  nullptr);
    const char* path = env->GetStringUTFChars(jpath, nullptr);

    if (!url || !path) {
        if (url)  env->ReleaseStringUTFChars(jurl, url);
        if (path) env->ReleaseStringUTFChars(jpath, path);
        log_to_file("download: invalid args");
        return env->NewStringUTF("invalid args");
    }

    {
        std::ostringstream ss;
        ss << "download: start url=" << url << " path=" << path;
        log_to_file(ss.str());
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        env->ReleaseStringUTFChars(jurl,  url);
        env->ReleaseStringUTFChars(jpath, path);
        log_to_file("download: curl init failed");
        return env->NewStringUTF("curl init failed");
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        env->ReleaseStringUTFChars(jurl,  url);
        env->ReleaseStringUTFChars(jpath, path);
        curl_easy_cleanup(curl);
        {
            std::ostringstream ss;
            ss << "download: file open failed path=" << path;
            log_to_file(ss.str());
        }
        return env->NewStringUTF("file open failed");
    }

    // Prepare progress callback data
    ProgressData pd;
    pd.last_percent = -1;
    pd.thiz_global = env->NewGlobalRef(thiz);
    pd.onProgressMethod = nullptr;

    // Try to get method ID for onDownloadProgress(int)
    jclass cls = env->GetObjectClass(thiz);
    if (cls) {
        pd.onProgressMethod = env->GetMethodID(cls, "onDownloadProgress", "(I)V");
        // cls is a local ref and will be released by JVM automatically when returning
    }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ofs);

    // Enable progress callbacks (use xferinfo if available)
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, xferinfo);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &pd);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    // ----------------------------------------------------
    // huggingface.co のときだけ SSL 検証を無効化
    // ----------------------------------------------------
    std::string surl(url);

    if (surl.rfind("https://huggingface.co/", 0) == 0) {
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        log_to_file("download: disabled SSL verification for huggingface.co");
    }

    curl_easy_setopt(curl, CURLOPT_USERAGENT,
        "Mozilla/5.0 (Linux; Android 14; Mobile) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Mobile Safari/537.36");

    CURLcode res = curl_easy_perform(curl);

    env->ReleaseStringUTFChars(jurl,  url);
    env->ReleaseStringUTFChars(jpath, path);
    curl_easy_cleanup(curl);
    ofs.close();

    if (pd.thiz_global) {
        env->DeleteGlobalRef(pd.thiz_global);
    }

    if (res != CURLE_OK) {
        std::ostringstream ss;
        ss << "download: curl download failed res=" << res << " msg=" << curl_easy_strerror(res);
        log_to_file(ss.str());
        return env->NewStringUTF("curl download failed");
    }

    log_to_file("download: ok");
    return env->NewStringUTF("ok");
}

// ---------------- JNI: init ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_init(
        JNIEnv *env, jobject /*thiz*/,
        jstring jModelPath
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    log_to_file("init: start");

    llama_jni_free();

    std::string model_path = jstring_to_std(env, jModelPath);

    {
        std::ostringstream ss;
        ss << "init: model_path=" << model_path;
        log_to_file(ss.str());
    }

    // store JavaVM for progress callback threads
    if (env->GetJavaVM(&g_jvm) != JNI_OK) {
        g_jvm = nullptr;
        log_to_file("init: GetJavaVM failed");
    } else {
        log_to_file("init: JavaVM stored");
    }

    // llama_backend_init requires a bool numa parameter in this header
    llama_backend_init(false);
    log_to_file("init: backend init");

    llama_model_params mparams = llama_model_default_params();
    // Note: function name in this header is llama_load_model_from_file
    g_model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!g_model) {
        log_to_file("init: failed to load model");
        return env->NewStringUTF("failed to load model");
    }
    log_to_file("init: model loaded");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = g_n_ctx;
    cparams.n_threads       = g_n_threads;
    cparams.n_batch         = g_n_batch;
    cparams.n_threads_batch = g_n_threads;

    // create context with model (API present in this header)
    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        llama_jni_free();
        log_to_file("init: failed to create context");
        return env->NewStringUTF("failed to create context");
    }

    // set RNG seed if desired
    llama_set_rng_seed(g_ctx, (uint32_t)LLAMA_DEFAULT_SEED);
    log_to_file("init: context created and RNG seed set");

    return env->NewStringUTF("ok");
}

// ---------------- JNI: generate ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_generate(
        JNIEnv *env, jobject /*thiz*/,
        jstring jPrompt
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_ctx || !g_model) {
        log_to_file("generate: not initialized");
        return env->NewStringUTF("not initialized");
    }

    std::string prompt = jstring_to_std(env, jPrompt);
    {
        std::ostringstream ss;
        ss << "generate: prompt_len=" << prompt.size();
        log_to_file(ss.str());
    }

    const int max_tokens = 128;

    // ---- KV キャッシュクリア ----
    // Use kv cache removal to clear all tokens: [0, inf)
    llama_kv_cache_tokens_rm(g_ctx, 0, -1);
    log_to_file("generate: kv cache cleared");

    // ---- トークナイズ（ヘッダ仕様）----
    std::vector<llama_token> tokens;
    tokens.resize(g_n_ctx);

    int32_t n_tokens = llama_tokenize(
            g_model,
            prompt.c_str(),
            (int)prompt.size(),
            tokens.data(),
            (int)tokens.size(),
            true
    );

    if (n_tokens <= 0) {
        log_to_file("generate: tokenize failed");
        return env->NewStringUTF("tokenize failed");
    }

    {
        std::ostringstream ss;
        ss << "generate: n_tokens=" << n_tokens;
        log_to_file(ss.str());
    }

    tokens.resize(n_tokens);

    int n_past = 0;
    std::string output;
    output.reserve(max_tokens * 4);

    // ---- プロンプト投入（1トークンずつ llama_eval を使用）----
    for (int i = 0; i < n_tokens; ++i) {
        llama_token tok = tokens[i];
        // llama_eval is available (deprecated in header but present)
        if (llama_eval(g_ctx, &tok, 1, n_past) != 0) {
            log_to_file("generate: eval failed (prompt)");
            return env->NewStringUTF("eval failed (prompt)");
        }
        ++n_past;

        // ログ：進捗（プロンプト投入）
        {
            std::ostringstream ss;
            ss << "generate: prompt token " << i << " id=" << (int)tok << " n_past=" << n_past;
            log_to_file(ss.str());
        }
    }

    // ---- 生成ループ ----
    for (int i = 0; i < max_tokens; ++i) {
        // After last eval, get logits for last token
        float * logits = llama_get_logits(g_ctx);
        if (!logits) {
            log_to_file("generate: no logits");
            return env->NewStringUTF("no logits");
        }

        const int n_vocab = llama_n_vocab(g_model);
        if (n_vocab <= 0) {
            log_to_file("generate: invalid vocab size");
            return env->NewStringUTF("invalid vocab size");
        }

        // build candidates
        std::vector<llama_token_data> cand_data;
        cand_data.resize((size_t)n_vocab);
        for (int t = 0; t < n_vocab; ++t) {
            cand_data[(size_t)t].id = (llama_token)t;
            cand_data[(size_t)t].logit = logits[t];
            cand_data[(size_t)t].p = 0.0f;
        }
        llama_token_data_array candidates = { cand_data.data(), (size_t)n_vocab, false };

        // apply sampling steps (softmax -> top_k -> top_p -> temp)
        llama_sample_softmax(g_ctx, &candidates);
        llama_sample_top_k(g_ctx, &candidates, g_top_k, 1);
        llama_sample_top_p(g_ctx, &candidates, g_top_p, 1);
        llama_sample_temp(g_ctx, &candidates, g_temp);

        // pick token
        llama_token id = llama_sample_token(g_ctx, &candidates);

        // check eos
        if (id == llama_token_eos(g_ctx)) {
            log_to_file("generate: reached EOS");
            break;
        }

        // token -> piece (ヘッダのシグニチャに合わせる)
        int32_t n_chars = llama_token_to_piece(
                g_model,
                id,
                nullptr,
                0
        );

        if (n_chars > 0) {
            std::string piece;
            piece.resize(n_chars);
            llama_token_to_piece(
                    g_model,
                    id,
                    piece.data(),
                    n_chars
            );
            output += piece;

            // ログ：生成トークン情報
            {
                std::ostringstream ss;
                ss << "generate: output token id=" << (int)id << " piece=\"" << piece << "\" i=" << i;
                log_to_file(ss.str());
            }
        } else {
            std::ostringstream ss;
            ss << "generate: token_to_piece returned n_chars=" << n_chars << " id=" << (int)id;
            log_to_file(ss.str());
        }

        // feed token into model for next step
        if (llama_eval(g_ctx, &id, 1, n_past) != 0) {
            log_to_file("generate: eval failed (generation)");
            return env->NewStringUTF("eval failed (generation)");
        }
        ++n_past;
    }

    {
        std::ostringstream ss;
        ss << "generate: finished, output_len=" << output.size();
        log_to_file(ss.str());
    }

    return env->NewStringUTF(output.c_str());
}

// ---------------- JNI: free ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_example_ollama_LlamaNative_free(
        JNIEnv *env, jobject /*thiz*/
) {
    log_to_file("Java_com_example_ollama_LlamaNative_free called");
    llama_jni_free();
}
