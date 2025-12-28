package com.example.PROJECTNAME;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LlamaNative llama = new LlamaNative();
        String result = llama.test("Hello JNI");

        TextView tv = new TextView(this);
        tv.setText(result);
        tv.setTextSize(24);

        setContentView(tv);
    }
}
