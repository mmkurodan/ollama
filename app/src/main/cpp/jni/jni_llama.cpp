#include <jni.h>
#include <string>
#include <vector>
#include <mutex>

#include <android/log.h>
#define LOG_TAG "LLAMA_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#include "llama.h"
#include <curl/curl.h>

// ---------------- グローバル ----------------
static std::mutex g_mutex;
static llama_model   *g_model   = nullptr;
static llama_context *g_ctx     = nullptr;
static llama_sampler *g_sampler = nullptr;

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
static size_t write_file(void *ptr, size_t size, size_t nmemb, void *stream) {
    FILE *fp = (FILE *)stream;
    return fwrite(ptr, size, nmemb, fp);
}

static std::string http_download(const std::string &url, const std::string &path) {
    CURL *curl = curl_easy_init();
    if (!curl) return "curl init failed";

    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp) return "file open failed";

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        return "download failed";
    }
    return "ok";
}

// ---------------- 解放 ----------------
static void llama_jni_free() {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_sampler) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }

    llama_backend_free();
}

// ---------------- JNI: download ----------------
extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_download(
        JNIEnv *env, jobject /*thiz*/,
        jstring jUrl,
        jstring jPath
) {
    std::string url  = jstring_to_std(env, jUrl);
    std::string path = jstring_to_std(env, jPath);

    std::string result = http_download(url, path);
    return env->NewStringUTF(result.c_str());
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

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    g_model = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!g_model) {
        return env->NewStringUTF("failed to load model");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = g_n_ctx;
    cparams.n_threads = g_n_threads;
    cparams.n_batch   = g_n_batch;

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        llama_jni_free();
        return env->NewStringUTF("failed to create context");
    }

    // ---- サンプラー作成 ----
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler *chain = llama_sampler_chain_init(chain_params);

    llama_sampler_chain_add(chain, llama_sampler_init_top_k(g_top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(g_top_p));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(g_temp));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(g_model));

    g_sampler = chain;

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

    if (!g_ctx || !g_model || !g_sampler) {
        return env->NewStringUTF("not initialized");
    }

    std::string prompt = jstring_to_std(env, jPrompt);
    int max_tokens = 128;

    // ---- トークナイズ ----
    std::vector<llama_token> tokens(g_n_ctx);

    int n_tokens = llama_tokenize(
            g_ctx,
            prompt.c_str(),
            prompt.size(),
            tokens.data(),
            tokens.size(),
            true
    );

    if (n_tokens <= 0) {
        return env->NewStringUTF("tokenize failed");
    }

    tokens.resize(n_tokens);

    // ---- batch ----
    llama_batch batch = llama_batch_init(g_n_batch, 0, 1);

    int n_past = 0;
    std::string output;
    output.reserve(max_tokens * 4);

    // ---- 初期 eval ----
    for (int i = 0; i < n_tokens; i++) {
        batch.n_tokens  = 1;
        batch.token[0]  = tokens[i];
        batch.pos[0]    = n_past;
        batch.seq_id[0] = 0;
        batch.logits[0] = false;

        if (llama_decode(g_ctx, batch) != 0) {
            llama_batch_free(batch);
            return env->NewStringUTF("decode failed");
        }
        n_past++;
    }

    // ---- 生成ループ ----
    for (int i = 0; i < max_tokens; i++) {

        llama_token id = llama_sampler_sample(g_sampler, g_ctx, nullptr, 0);

        if (id == llama_token_eos(g_ctx)) break;

        // ---- トークン → 文字列 ----
        int n_chars = llama_token_to_piece(
                g_ctx, id,
                nullptr, 0,
                false, false
        );

        if (n_chars > 0) {
            std::string piece;
            piece.resize(n_chars);

            llama_token_to_piece(
                    g_ctx, id,
                    piece.data(), n_chars,
                    false, false
            );

            output += piece;
        }

        // ---- 次のトークンをモデルに入力 ----
        batch.n_tokens  = 1;
        batch.token[0]  = id;
        batch.pos[0]    = n_past;
        batch.seq_id[0] = 0;
        batch.logits[0] = false;

        if (llama_decode(g_ctx, batch) != 0) {
            llama_batch_free(batch);
            return env->NewStringUTF("decode failed");
        }

        n_past++;
    }

    llama_batch_free(batch);

    return env->NewStringUTF(output.c_str());
}
