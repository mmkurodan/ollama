#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <fstream>

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

// 設定
static int   g_n_ctx      = 512;
static int   g_n_threads  = 2;
static int   g_n_batch    = 16;
static float g_temp       = 0.7f;
static float g_top_p      = 0.9f;
static int   g_top_k      = 40;

// ---------------- ユーティリティ ----------------
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

    if (attached) {
        g_jvm->DetachCurrentThread();
    }

    return 0;
}

// ---------------- 解放 ----------------
static void llama_jni_free() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ctx) {
        // free context
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        // free model
        llama_free_model(g_model);
        g_model = nullptr;
    }

    // backend cleanup
    llama_backend_free();
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
        return env->NewStringUTF("invalid args");
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        env->ReleaseStringUTFChars(jurl,  url);
        env->ReleaseStringUTFChars(jpath, path);
        return env->NewStringUTF("curl init failed");
    }

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        env->ReleaseStringUTFChars(jurl,  url);
        env->ReleaseStringUTFChars(jpath, path);
        curl_easy_cleanup(curl);
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
        return env->NewStringUTF("curl download failed");
    }

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

    llama_jni_free();

    std::string model_path = jstring_to_std(env, jModelPath);

    // store JavaVM for progress callback threads
    if (env->GetJavaVM(&g_jvm) != JNI_OK) {
        g_jvm = nullptr;
    }

    // llama_backend_init requires a bool numa parameter in this header
    llama_backend_init(false);

    llama_model_params mparams = llama_model_default_params();
    // Note: function name in this header is llama_load_model_from_file
    g_model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!g_model) {
        return env->NewStringUTF("failed to load model");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = g_n_ctx;
    cparams.n_threads       = g_n_threads;
    cparams.n_batch         = g_n_batch;
    cparams.n_threads_batch = g_n_threads;

    // create context with model (API present in this header)
    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        llama_jni_free();
        return env->NewStringUTF("failed to create context");
    }

    // set RNG seed if desired
    llama_set_rng_seed(g_ctx, (uint32_t)LLAMA_DEFAULT_SEED);

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
        return env->NewStringUTF("not initialized");
    }

    std::string prompt = jstring_to_std(env, jPrompt);
    const int max_tokens = 128;

    // ---- KV キャッシュクリア ----
    // Use kv cache removal to clear all tokens: [0, inf)
    llama_kv_cache_tokens_rm(g_ctx, 0, -1);

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
        return env->NewStringUTF("tokenize failed");
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
            return env->NewStringUTF("eval failed (prompt)");
        }
        ++n_past;
    }

    // ---- 生成ループ ----
    for (int i = 0; i < max_tokens; ++i) {
        // After last eval, get logits for last token
        float * logits = llama_get_logits(g_ctx);
        if (!logits) {
            return env->NewStringUTF("no logits");
        }

        const int n_vocab = llama_n_vocab(g_model);
        if (n_vocab <= 0) {
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
        }

        // feed token into model for next step
        if (llama_eval(g_ctx, &id, 1, n_past) != 0) {
            return env->NewStringUTF("eval failed (generation)");
        }
        ++n_past;
    }

    return env->NewStringUTF(output.c_str());
}

// ---------------- JNI: free ----------------
extern "C"
JNIEXPORT void JNICALL
Java_com_example_ollama_LlamaNative_free(
        JNIEnv *env, jobject /*thiz*/
) {
    llama_jni_free();
}
