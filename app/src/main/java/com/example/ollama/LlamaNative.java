package com.example.PROJECTNAME;

public class LlamaNative {

    static {
        System.loadLibrary("llama_jni");
    }

    public native String test(String prompt);
}
