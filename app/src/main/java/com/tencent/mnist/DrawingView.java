package com.tencent.mnist;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;

public class DrawingView extends View {
    private ArrayList<Path> mPaths;
    private Path mPath;
    private Paint mPaint;
    private float mX, mY;
    private static final float TOUCH_TOLERANCE = 4;

    public DrawingView(Context context, AttributeSet attr) {
        super(context, attr);
        setFocusable(true);
        setFocusableInTouchMode(true);
        setBackgroundColor(Color.CYAN);
        onCanvasInitialization();
    }

    public void onCanvasInitialization() {
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setDither(true);
        mPaint.setColor(Color.BLACK);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setStrokeWidth(2);

        mPaths = new ArrayList<>();

    }
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touch_start(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                touch_move(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                touch_up();
                invalidate();
                break;
        }
        return true;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        for (Path p : mPaths) {
            canvas.drawPath(p, mPaint);
        }
    }
    private void touch_start(float x, float y) {
        mX = x;
        mY = y;
        mPath = new Path();
        mPath.moveTo(x,y);
        mPaths.add(mPath);
    }
    private void touch_move(float x, float y) {
        float dx = Math.abs(x - mX);
        float dy = Math.abs(y - mY);
        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            mPath.quadTo(mX, mY, (x + mX) / 2, (y + mY) / 2);
            mX = x;
            mY = y;
        }
    }
    private void touch_up() {
        mPath.lineTo(mX, mY);
    }

    public void reset()
    {
        Log.d("fgt","clear");
        mPaths.clear();
        invalidate();
    }
    static final int MNIST_WIDTH = 28;
    static final int MNIST_HEIGHT = 28;
    private int[] pixels = new int[MNIST_WIDTH*MNIST_HEIGHT];
    private float[] floatValues = new float[MNIST_WIDTH*MNIST_HEIGHT];
    private float[] outputs = new float[10];
    public void inference(Context context,Classifier cls){
        Bitmap imageBitmap = Bitmap.createBitmap(MNIST_WIDTH,MNIST_HEIGHT, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(imageBitmap);
        canvas.drawColor(Color.WHITE);
        Matrix scaleMatrix = new Matrix();
        scaleMatrix.setScale((float)MNIST_WIDTH/getWidth(),(float)MNIST_HEIGHT/getHeight());
        for(Path p:mPaths){
            Path sp = new Path(p);
            sp.transform(scaleMatrix);
            canvas.drawPath(sp,mPaint);
        }
        imageBitmap.getPixels(pixels, 0, MNIST_WIDTH, 0, 0, MNIST_WIDTH, MNIST_HEIGHT);
        for (int i = 0; i < pixels.length; ++i) {
            final int val = pixels[i];
            floatValues[i] = (float)(((val) & 0xFF)) / 255;
        }

        cls.run(floatValues, outputs);
        for(int i=0;i<outputs.length;++i){
            Log.d("fgt","index:"+i+"  possibility:"+outputs[i]);
        }
        if(context != null)
            CapturePhotoUtils.insertImage(context.getContentResolver(),imageBitmap,"mnist","this is minst draw");
    }
}