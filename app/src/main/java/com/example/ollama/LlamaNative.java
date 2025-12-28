package com.example.ollama;

public class LlamaNative {

    static {
        System.loadLibrary("llama_jni");
    }

    public native String download(String url, String path);
    public native String init(String modelPath);
}
