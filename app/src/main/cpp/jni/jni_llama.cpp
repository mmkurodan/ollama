#include <jni.h>
#include <string>
#include <fstream>
#include <curl/curl.h>

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

    std::string msg = std::string("init() called with: ") + path;

    env->ReleaseStringUTFChars(jpath, path);
    return env->NewStringUTF(msg.c_str());
}
