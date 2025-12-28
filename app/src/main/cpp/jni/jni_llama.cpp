#include <jni.h>
#include <string>
#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "LlamaJNI", __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_PROJECTNAME_LlamaNative_test(JNIEnv* env, jobject thiz, jstring jPrompt) {
    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string out = std::string("JNI OK: ") + prompt;
    env->ReleaseStringUTFChars(jPrompt, prompt);
    return env->NewStringUTF(out.c_str());
}
