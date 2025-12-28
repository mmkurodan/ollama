#include <jni.h>
#include <string>
#include <fstream>
#include <vector>
#include <curl/curl.h>

#include "llama.h"

static llama_context* g_ctx = nullptr;

static size_t write_data(void* ptr, size_t size, size_t nmemb, void* userdata) {
    std::ofstream* ofs = reinterpret_cast<std::ofstream*>(userdata);
    ofs->write(reinterpret_cast<const char*>(ptr), size * nmemb);
    return size * nmemb;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_download(
        JNIEnv* env,
        jobject /*thiz*/,
        jstring jurl,
        jstring jpath) {

    const char* url  = env->GetStringUTFChars(jurl,  nullptr);
    const char* path = env->GetStringUTFChars(jpath, nullptr);

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

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ofs);

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

    if (res != CURLE_OK) {
        return env->NewStringUTF("curl download failed");
    }

    return env->NewStringUTF("ok");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_init(
        JNIEnv* env,
        jobject /*thiz*/,
        jstring jpath) {

    const char* path = env->GetStringUTFChars(jpath, nullptr);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();

    llama_model* model = llama_load_model_from_file(path, model_params);
    if (!model) {
        env->ReleaseStringUTFChars(jpath, path);
        return env->NewStringUTF("model load failed");
    }

    g_ctx = llama_new_context_with_model(model, ctx_params);
    if (!g_ctx) {
        env->ReleaseStringUTFChars(jpath, path);
        return env->NewStringUTF("context init failed");
    }

    env->ReleaseStringUTFChars(jpath, path);
    return env->NewStringUTF("model loaded");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_ollama_LlamaNative_generate(
        JNIEnv* env,
        jobject /*thiz*/,
        jstring jprompt) {

    if (!g_ctx) {
        return env->NewStringUTF("context not initialized");
    }

    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);

    // トークン化
    std::vector<llama_token> tokens =
        llama_tokenize(g_ctx, prompt, true);

    // プロンプト入力
    for (auto t : tokens) {
        llama_eval(g_ctx, &t, 1, 0, 0);
    }

    // 生成ループ
    std::string output;
    for (int i = 0; i < 64; i++) {
        llama_token id = llama_sample_token(g_ctx);
        if (id == llama_token_eos(g_ctx)) break;

        const char* piece = llama_token_to_str(g_ctx, id);
        output += piece;

        llama_eval(g_ctx, &id, 1, 0, 0);
    }

    env->ReleaseStringUTFChars(jprompt, prompt);
    return env->NewStringUTF(output.c_str());
}
