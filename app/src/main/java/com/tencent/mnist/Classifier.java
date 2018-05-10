package com.tencent.mnist;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.channels.FileChannel;

public class Classifier {
    private Interpreter mTfLite;
    public Classifier(Context context) {
        AssetManager assetManager = context.getAssets();

        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd("mnist.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            mTfLite = new Interpreter(fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    public void run(float[][][][] in,float[][] out){
        long startTime;
        long endTime;
        startTime = SystemClock.uptimeMillis();
        mTfLite.run(in, out);
        endTime = SystemClock.uptimeMillis();
        Log.i("fgt", "Inf time: " + (endTime - startTime));
    }
}
