package com.tencent.mnist;

import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Environment;
import android.support.v7.app.AlertDialog;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;

public class DrawingView extends View {
    private ArrayList<Path> mPaths;
    private Path mPath;
    private Paint mPaint;
    private float mX, mY;
    private float mXmin = Float.POSITIVE_INFINITY,mXmax = Float.NEGATIVE_INFINITY,
            mYmin = Float.POSITIVE_INFINITY,mYmax = Float.NEGATIVE_INFINITY;
    private static final float TOUCH_TOLERANCE = 4;
    private static int SAVE_INDEX = 1;

    public DrawingView(Context context, AttributeSet attr) {
        super(context, attr);
        setFocusable(true);
        setFocusableInTouchMode(true);
        setBackgroundColor(Color.GRAY);
        onCanvasInitialization();
    }

    public void onCanvasInitialization() {
        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        //mPaint.setDither(true);
        mPaint.setColor(Color.WHITE);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.MITER);
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setStrokeWidth(3);

        mPaths = new ArrayList<>();

    }
    private void updateXyRange(float x,float y){
        //Log.d("fgt","updateXyRange x:"+x+" y:"+y);
        if(x<mXmin){
            mXmin = x;
        }
        if(x>mXmax){
            mXmax = x;
        }
        if(y<mYmin){
            mYmin = y;
        }
        if(y>mYmax){
            mYmax = y;
        }
        //Log.d("fgt","updateXyRange minx:"+mXmin+" maxx:"+mXmax+" miny:"+mYmin+" maxy:"+mYmax);
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
            updateXyRange(x,y);
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
        mXmin = Float.POSITIVE_INFINITY;
        mXmax = Float.NEGATIVE_INFINITY;
        mYmin = Float.POSITIVE_INFINITY;
        mYmax = Float.NEGATIVE_INFINITY;
        mPaths.clear();
        invalidate();
    }
    static final int MNIST_WIDTH = 28;
    static final int MNIST_HEIGHT = 28;
    private int[] pixels = new int[MNIST_WIDTH*MNIST_HEIGHT];
    private float[][][][] floatValues = new float[1][MNIST_WIDTH][MNIST_HEIGHT][1];
    private float[][] outputs = new float[1][10];
    public void inference(Context context,Classifier cls){
        Bitmap imageBitmap = Bitmap.createBitmap(MNIST_WIDTH,MNIST_HEIGHT, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(imageBitmap);
        canvas.drawColor(Color.BLACK);
        Matrix scaleMatrix = new Matrix();

        float xr = mXmax - mXmin;
        float yr = mYmax - mYmin;
        float range = 0;
        float xtrans = 0;
        float ytrans = 0;
        if(xr>yr){
            range = xr;
            xtrans = -mXmin;
            ytrans = -mYmin+(xr-yr)/2;
        }else{
            range = yr;
            xtrans = -mXmin+(yr-xr)/2;
            ytrans = -mYmin;
        }
        Log.d("fgt","width:"+getWidth()+" height:"+getHeight());
        Log.d("fgt","minx:"+mXmin+" maxx:"+mXmax+" miny:"+mYmin+" maxy:"+mYmax);
        Log.d("fgt","xtrans:"+xtrans+" ytrans:"+ytrans);
        scaleMatrix.setTranslate (xtrans,ytrans);
        Log.d("fgt","range:"+range);
        float scale = 20f/range;
        scaleMatrix.postScale(scale,scale);
        scaleMatrix.postTranslate (4,4);
        for(Path p:mPaths){
            Path sp = new Path(p);
            sp.transform(scaleMatrix);
            canvas.drawPath(sp,mPaint);
        }
        imageBitmap.getPixels(pixels, 0, MNIST_WIDTH, 0, 0, MNIST_WIDTH, MNIST_HEIGHT);
        for (int i = 0; i < MNIST_WIDTH; ++i) {
            for(int j=0;j< MNIST_HEIGHT;++j) {
                final int val = pixels[i * 28 + j];
                floatValues[0][i][j][0] = (((val) & 0xFF) - 255.0f/2) / 255.0f;
            }
        }

        cls.run(floatValues, outputs);
        int maxIndex = 0;
        for(int i=0;i<outputs[0].length;++i){
            if(outputs[0][maxIndex] < outputs[0][i]) {
                maxIndex = i;
            }

            Log.d("fgt","index:"+i+"  possibility:"+outputs[0][i]);
        }
        AlertDialog alertDialog = new AlertDialog.Builder(context).create();
        alertDialog.setTitle("recognize result");
        alertDialog.setMessage(maxIndex+" probability:"+outputs[0][maxIndex]);
        alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                        reset();
                    }
                });
        alertDialog.show();
//        if(context != null)
//            saveImageToExternal(context,""+SAVE_INDEX++,imageBitmap);
    }
    private void saveImageToExternal(Context context,String imgName, Bitmap bm) {
//Create Path to save Image
        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES); //Creates app specific folder
        path.mkdirs();
        File imageFile = new File(path, imgName + ".png"); // Imagename.png
        try {
            FileOutputStream out = new FileOutputStream(imageFile);
            bm.compress(Bitmap.CompressFormat.PNG, 100, out); // Compress Image
            out.flush();
            out.close();
            MediaScannerConnection.scanFile(context, new String[]{imageFile.getAbsolutePath()}, null, new MediaScannerConnection.OnScanCompletedListener() {
                public void onScanCompleted(String path, Uri uri) {
                    Log.i("fgt", "Scanned " + path + ":");
                    Log.i("fgt", "-> uri=" + uri);
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}