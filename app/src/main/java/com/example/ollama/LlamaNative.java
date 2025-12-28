package com.example.ollama;

public class LlamaNative {

    static {
        System.loadLibrary("llama_jni");  // ← これが正しい
    }

    public native String init(String modelPath);

    public native String generate(String prompt);
}
