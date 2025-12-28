package com.example.ollama;

public class LlamaNative {

    static {
        // libllama.so をロード（CMake で生成される想定のライブラリ名に合わせる）
        System.loadLibrary("llama");
    }

    // モデル初期化（ここに modelPath が渡る）
    public native String init(String modelPath);

    // テキスト生成（まだダミーでOK）
    public native String generate(String prompt);
}
