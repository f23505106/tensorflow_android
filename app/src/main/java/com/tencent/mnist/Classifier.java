package com.tencent.mnist;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.channels.FileChannel;
import java.util.Arrays;

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
            Log.d("fgt","InputTensorCount:"+mTfLite.getInputTensorCount()+"  OutputTensorCount:"+mTfLite.getOutputTensorCount());
            Log.d("fgt","inputTensor:"+mTfLite.getInputTensor(0).dataType());
            Log.d("fgt","inputTensor shape:"+Arrays.toString(mTfLite.getInputTensor(0).shape()));
            Log.d("fgt","OutputTensor0:"+mTfLite.getOutputTensor(0).dataType());
            Log.d("fgt","OutputTensor1:"+mTfLite.getOutputTensor(1).dataType());
            Log.d("fgt","OutputTensor shape0:"+Arrays.toString(mTfLite.getOutputTensor(0).shape()));
            Log.d("fgt","OutputTensor shape1:"+Arrays.toString(mTfLite.getOutputTensor(1).shape()));
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
