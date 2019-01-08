/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.demo.tracking;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
import android.graphics.Path;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Pair;
import android.util.TypedValue;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.tensorflow.demo.Classifier.Recognition;
import org.tensorflow.demo.R;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.initializer.ObjectReferenceList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Vector;

import static org.tensorflow.demo.MrCameraActivity.MIN_MATCH_COUNT;

/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */

public class DemoMultiBoxTrackerWithARCore {

    private int minMatchCount = Math.round(MIN_MATCH_COUNT*10/30);

    private final Logger logger = new Logger();

    private static final float TEXT_SIZE_DIP = 18;

    private static final int TRACKING_TIMEOUT = 1000;

    private static final int TRACKING_DIFFERENCE = 100;

    // Maximum percentage of a box that can be overlapped by another box at detection time. Otherwise
    // the lower scored box (new or old) will be removed.
    private static final float MAX_OVERLAP = 0.2f;

    private static final float MIN_SIZE = 16.0f;

    private static int privilegeLevel;

    // Allow replacement of the tracked box with new results if
    // correlation has dropped below this level.
    private static final float MARGINAL_CORRELATION = 0.75f;

    // Consider object to be lost if correlation falls below this threshold.
    private static final float MIN_CORRELATION = 0.3f;

    private static final int[] COLORS = {
        Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.WHITE,
        Color.parseColor("#55FF55"), Color.parseColor("#FFA500"), Color.parseColor("#FF8888"),
        Color.parseColor("#AAAAFF"), Color.parseColor("#FFFFAA"), Color.parseColor("#55AAAA"),
        Color.parseColor("#AA33AA"), Color.parseColor("#0D0068")
    };

    static final String[] sensitiveObjects = new String[]
          {"person", "bed", "toilet", "laptop", "cell phone"}; //high sensitivity objects

    private final Queue<Integer> availableColors = new LinkedList<Integer>();

    public ObjectTracker objectTracker;

    final List<Pair<Float, RectF>> screenRects = new LinkedList<Pair<Float, RectF>>();

    /*// This is the reference frame for tracking.
    private Bitmap prevFrame;*/

    private static class TrackedRecognition {
        ObjectTracker.TrackedObject trackedObject;
        RectF location;
        float detectionConfidence;
        int color;
        String title;
        long lastUpdate;
        int coverColor;
        boolean sensitivity = false;
        Path pathLocation = new Path();
    }

    private final List<TrackedRecognition> trackedObjects = new LinkedList<TrackedRecognition>();

    private static class CvTrackedRecognition {
        TrackedRecognition trackedRecognition;
        RectF location;
        float detectionConfidence;
        int color;
        String title;
        Mat RefImageMat;
    }

    private static class RefFrame {
        private static final RefFrame instance = new RefFrame();
        Bitmap refFrame;
        Mat refMat;
        MatOfPoint refPoints;

        private RefFrame(){ }

        public static RefFrame getInstance() {
          return instance;
        }

    }

    private final List<CvTrackedRecognition> cvTrackedObjects = new LinkedList<>();

    private final Paint boxPaint = new Paint();

    private final float textSizePx;
    private final BorderedText borderedText;

    private Matrix frameToCanvasMatrix;

    private int frameWidth;
    private int frameHeight;

    private int sensorOrientation;
    private Context context;

    private static ObjectReferenceList objectReferenceList = ObjectReferenceList.getInstance();

    private static Resources res;

    private Bitmap coverBitmap;

    public DemoMultiBoxTrackerWithARCore(final Context context) {
        this.context = context;
        for (final int color : COLORS) {
            availableColors.add(color);
        }

        boxPaint.setColor(Color.RED);
        boxPaint.setStyle(Style.STROKE);
        boxPaint.setStrokeWidth(12.0f);
        boxPaint.setStrokeCap(Cap.ROUND);
        boxPaint.setStrokeJoin(Join.ROUND);
        boxPaint.setStrokeMiter(100);

        textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, context.getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);

        res = context.getResources();

        coverBitmap = BitmapFactory.decodeResource(res, R.drawable.csiro);

    }

    public void setPrivilege(int privilege){
        privilegeLevel = privilege;
    }

    private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
    }

//    public synchronized void drawDebug(final Canvas canvas) {
//        logger.i("Entered drawDebug");
//        final Paint textPaint = new Paint();
//        textPaint.setColor(Color.WHITE);
//        textPaint.setTextSize(60.0f);
//
//        final Paint boxPaint = new Paint();
//        boxPaint.setColor(Color.RED);
//        boxPaint.setAlpha(200);
//        boxPaint.setStyle(Style.STROKE);
//
//        for (final Pair<Float, RectF> detection : screenRects) {
//          final RectF rect = detection.second;
//          canvas.drawRect(rect, boxPaint);
//          canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
//          borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
//        }
//
//        if (objectTracker == null) {
//          logger.i("DrawDebug: Object Tracker is null.");
//          return;
//        }
//
//        // Draw correlations.
//        for (final TrackedRecognition recognition : trackedObjects) {
//          final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;
//
//          final RectF trackedPos = trackedObject.getTrackedPositionInPreviewFrame();
//
//          if (getFrameToCanvasMatrix().mapRect(trackedPos)) {
//            final String labelString = String.format("%.2f", trackedObject.getCurrentCorrelation());
//            borderedText.drawText(canvas, trackedPos.right, trackedPos.bottom, labelString);
//          }
//        }
//
//        final Matrix matrix = getFrameToCanvasMatrix();
//        objectTracker.drawDebug(canvas, matrix);
//    }

    public synchronized void trackResults(
        final List<Recognition> results, final byte[] frame, final long timestamp) {

        logger.i("Processing %d results from %d", results.size(), timestamp);
        processResults(timestamp, results, frame);
  }

  public synchronized void draw(final Canvas canvas) {

      RefFrame referenceFrame = RefFrame.getInstance();

      final boolean rotated = sensorOrientation % 180 == 90;
      final float multiplier =
        Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                 canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));

      logger.i("Tracker: (WxH) Preview dims: %dx%d; Canvas dims: %dx%d",
              frameWidth,frameHeight,
              canvas.getWidth(), canvas.getHeight());

      frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false); // originally false

    for (final TrackedRecognition recognition : trackedObjects) {
      final RectF trackedPos =
          (objectTracker != null)
              ? recognition.trackedObject.getTrackedPositionInPreviewFrame()
              : new RectF(recognition.location);

      getFrameToCanvasMatrix().mapRect(trackedPos);
      boxPaint.setColor(recognition.color);

      final float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;

      logger.i("Drawer: %s, Sensitivity is %s", recognition.title, String.valueOf(recognition.sensitivity));

      if (recognition.sensitivity) {

          Rect roundedLocation = new Rect();
          trackedPos.round(roundedLocation);

          final int coverColor = referenceFrame
                  .refFrame
                  .getPixel(Math.min(Math.abs(roundedLocation.centerX()),frameWidth/2),
                          Math.min(Math.abs(roundedLocation.centerY()),frameHeight/2));

          //logger.i("Color: %d ", coverColor);

          Paint paint = new Paint(); // Fill
          paint.setStyle(Style.FILL);
          paint.setColor(coverColor);
          paint.setAlpha(250); // higher number is opaque, 127 is too transparent.

/*          final Paint paint = new Paint(); // red box
          paint.setColor(Color.RED);
          paint.setStyle(Style.STROKE);
          paint.setStrokeWidth(2.0f);*/

          if (recognition.pathLocation.isEmpty()) {
              // expanded the destination Rect to prevent leaks
              final RectF secretPos = new RectF(
                      0.9f*trackedPos.left,
                      1.1f*trackedPos.top,
                      1.1f*trackedPos.right,
                      0.9f*trackedPos.bottom);

/*        canvas.drawBitmap(coverBitmap,
                new Rect(0,0,coverBitmap.getWidth(),coverBitmap.getHeight()),
                trackedPos,
                boxPaint);*/

              //float cornerSize = Math.min(secretPos.width(), secretPos.height()) / 16.0f;
              canvas.drawRect(trackedPos, paint);    // fill
          } else {
              logger.i("Path is drawn.");
              canvas.drawPath(recognition.pathLocation, paint);
          }

      } else {

        canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

        final String labelString =
                !TextUtils.isEmpty(recognition.title)
                        ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                        : String.format("%.2f", recognition.detectionConfidence);
        borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
      }
    }
  }

//  private Paint sPaint;

    /**
     * This is the color management code for the sanitized Paint so that the color does not change
     * as fast as the new frame just to make it look nicer.
     */

    private static class ColorManagement {
        Integer prevColor;
        int redP;
        int greenP;
        int blueP;

        private ColorManagement(){
        }

        public void setPrevColor(int color){
            this.prevColor = color;
            this.redP = Color.red(color);
            this.greenP = Color.green(color);
            this.blueP = Color.blue(color);
        }

        public int getNewColor(int color){
            int redC = Color.red(color);
            int greenC = Color.green(color);
            int blueC = Color.blue(color);

            int redD = (redC - this.redP) % 8;
            int greenD = (greenC - this.greenP) % 8;
            int blueD = (blueC - this.blueP) % 8;

            final int finalColor = Color.rgb(this.redP + redD,
                    this.greenP + greenD,
                    this.blueP + blueD);

            return finalColor;
        }

    }

    private static ColorManagement colorManager = new ColorManagement();

    public synchronized void drawSanitized(final Canvas canvas) {

        RefFrame referenceFrame = RefFrame.getInstance();

        final boolean rotated = sensorOrientation % 180 == 90;
        final float multiplier =
                Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                        canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));

        logger.i("Tracker: (WxH) Preview dims: %dx%d; Canvas dims: %dx%d",
                frameWidth,frameHeight,
                canvas.getWidth(), canvas.getHeight());

        frameToCanvasMatrix =
                ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        canvas.getWidth(),
                        canvas.getHeight(),
                        sensorOrientation,
                        false);
/*                ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                        (int) (multiplier * (rotated ? frameWidth : frameHeight)),
                        sensorOrientation,
                        false); // originally false*/

        final int centerColor = referenceFrame
                .refFrame
                .getPixel(frameWidth/2,frameHeight/2);

        if (colorManager.prevColor == null) colorManager.setPrevColor(centerColor);

        final int finalColor = colorManager.getNewColor(centerColor);
        colorManager.setPrevColor(finalColor);

        //Draw Overlay
        Paint sPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        sPaint.setColor(finalColor);
        sPaint.setStyle(Paint.Style.FILL);
        sPaint.setAlpha(250);

        //Draw transparent shape
        //sPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        //canvas.drawRoundRect(circleRect, radius, radius, sPaint);

        if (privilegeLevel==0) {
            sPaint.setAlpha(255);
            canvas.drawPaint(sPaint);
        } else if (privilegeLevel ==1) {
            sPaint.setAlpha(250);
            canvas.drawPaint(sPaint);
        } else if (privilegeLevel ==2) {
            sPaint.setAlpha(240);
            canvas.drawPaint(sPaint);
            //Draw transparent shape
            sPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        } else if (privilegeLevel ==3) {
            sPaint.setAlpha(230);
            canvas.drawPaint(sPaint);
            //Draw transparent shape
            sPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
        }

        int ypos = 25;
        for (final TrackedRecognition recognition : trackedObjects) {
            final RectF trackedPos =
                    (objectTracker != null)
                            ? recognition.trackedObject.getTrackedPositionInPreviewFrame()
                            : new RectF(recognition.location);

            getFrameToCanvasMatrix().mapRect(trackedPos);
            boxPaint.setColor(recognition.color);

            final float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;

            logger.i("Privilege level = %d", privilegeLevel);
            if (recognition.sensitivity) {
                //Do not draw if sensitive.
                continue;
            }

            if (privilegeLevel==0) {
                ypos += 50;
                final String labelString =
                        !TextUtils.isEmpty(recognition.title)
                                ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                                : String.format("%.2f", recognition.detectionConfidence);
                borderedText.drawText(canvas, 50, ypos, labelString);
            } else if (privilegeLevel==1) {
                //sPaint.setAlpha(250);
                //canvas.drawPaint(sPaint);
                // centroid
                if (!recognition.sensitivity) {
/*                    Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
                    fillPaint.setColor(recognition.color);
                    fillPaint.setStyle(Paint.Style.FILL);*/

                    String drawableName = recognition.title;
                    drawableName = drawableName.replace(" ","_");

                    int drawableId = res.getIdentifier(drawableName, "drawable", context.getPackageName());

                    logger.i(String.format("Drawables: %d: %s",drawableId,drawableName));

                    if (drawableId==0) continue;

                    Bitmap representationBitmap = BitmapFactory.decodeResource(res, drawableId);
                    canvas.drawBitmap(representationBitmap,
                            new Rect(0,0,representationBitmap.getWidth(),representationBitmap.getHeight()),
                            trackedPos,
                            sPaint);

                    final String labelString =
                            !TextUtils.isEmpty(recognition.title)
                                    ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                                    : String.format("%.2f", recognition.detectionConfidence);
                    borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.centerY() + cornerSize, labelString);
                }
            } else if (privilegeLevel==2) {
/*                sPaint.setAlpha(250);
                canvas.drawPaint(sPaint);
                //Draw transparent shape
                sPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));*/

                if (!recognition.sensitivity) {

                    canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

                    final String labelString =
                            !TextUtils.isEmpty(recognition.title)
                                    ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                                    : String.format("%.2f", recognition.detectionConfidence);
                    borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
                }
            } else if (privilegeLevel==3) {
/*                canvas.drawPaint(sPaint);
                //Draw transparent shape
                sPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));*/

                if (!recognition.sensitivity) {

                    // If object is bigger than the canvas dimensions, do not print it.
                    if (trackedPos.width()*trackedPos.height() > canvas.getWidth()*canvas.getHeight()) continue;

                    canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, sPaint);

                    final String labelString =
                            !TextUtils.isEmpty(recognition.title)
                                    ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                                    : String.format("%.2f", recognition.detectionConfidence);
                    borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
                }
            } else {
                canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

                    final String labelString =
                            !TextUtils.isEmpty(recognition.title)
                                    ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                                    : String.format("%.2f", recognition.detectionConfidence);
                    borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
            }

        }
    }

    private void refreshTrackedObjects(Bitmap frame){

    Iterator<TrackedRecognition> iterator = trackedObjects.iterator();

    while (iterator.hasNext()) {
      TrackedRecognition recognition = iterator.next();
      if (SystemClock.uptimeMillis() - recognition.lastUpdate > TRACKING_TIMEOUT) {
        iterator.remove();
        // Remove tracked objects if last update was more than a second ago.
        continue;
      }

      Rect roundedLocation = new Rect();
      recognition.location.round(roundedLocation);

      try {
        recognition.sensitivity = objectReferenceList.isSensitive(recognition.title);
      } catch (Exception e) {
        e.printStackTrace();
      }

    }

  }

  private boolean initialized = false;
/*
  public void setInitialize(boolean set){
    initialized = set;
  }*/

  public synchronized void onFrame(
      final int w,
      final int h,
      final int rowStride,
      final int sensorOrienation,
      final byte[] frame,
      final long timestamp) {

    if (objectTracker == null && !initialized) {
      ObjectTracker.clearInstance();

      logger.i("Initializing ObjectTracker: %dx%d", w, h);
      objectTracker = ObjectTracker.getInstance(w, h, rowStride, true);
      frameWidth = w;
      frameHeight = h;
      this.sensorOrientation = sensorOrienation;
      initialized = true;

      if (objectTracker == null) {
        String message =
            "Object tracking support not found. "
                + "See tensorflow/examples/android/README.md for details.";
        Toast.makeText(context, message, Toast.LENGTH_LONG).show();
        logger.e(message);
      }
    }

    if (objectTracker == null) {
      logger.i("OnFrame: Object Tracker is null.");
      return;
    }

    objectTracker.nextFrame(frame, null, timestamp, null, true);

    // Clean up any objects not worth tracking any more.
    final LinkedList<TrackedRecognition> copyList = new LinkedList<>(trackedObjects);

    for (final TrackedRecognition recognition : copyList) {
      final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;
      final float correlation = trackedObject.getCurrentCorrelation();
      if (correlation < MIN_CORRELATION) {
        logger.v("Removing tracked object %s because NCC is %.2f", trackedObject, correlation);
        trackedObject.stopTracking();
        trackedObjects.remove(recognition);

        availableColors.add(recognition.color);
      }
    }
  }

    public synchronized void HomogFrameTracker(
            final int w,
            final int h,
            final int sensorOrientation,
            final Bitmap frame,
            final long timestamp) {

        if (trackedObjects.isEmpty()) {
            logger.i("Nothing new to track.");
            return;
        }

/*        frameWidth = w;
        frameHeight = h;*/

        // Update time and sensitivity, i.e. check for timeouts
        refreshTrackedObjects(frame);


//        if (!initialized) {
//            // This if statement waits for detections to arrive from the detectors.
//            //cvTrackedObjects.clear();

//            logger.i("%d, Initializing Homog-CV-Tracker: %dx%d", timestamp, w, h);

        frameWidth = w;
        frameHeight = h;
        this.sensorOrientation = sensorOrientation;
//            initialized = true;
//            return;
//        }


        //trackedObjects.clear(); // Does not have to be cleared everytime.

        long start = System.currentTimeMillis();

        homogBasedTracker(timestamp, frame);

        long end = System.currentTimeMillis() - start;
        logger.i("CV (Homography-based) Frame tracking time: %d", end);

    }

    /**
     * Experimenting on Homography-based tracking instead of using orb for
     * every reference object.

     Based from these steps:

     use cv2.goodFeaturesToTrack to find good corners.
     use cv2.calcOpticalFlowPyrLK to track the corners.
     use cv2.findHomography to compute the homography matrix.
     use cv2.warpPerspective to transform video frame.

     Here are the general steps:"
     1. The RectF location results from the detection are kept at the beginning
        of the tracking sequence while another detection is being done.
     2. New frames arrive at the tracker and we compute homography transformations
        with the last frame from the detections as reference.
     3. The current detection locations are transformed using these frame-by-frame
        transformations.

     This approach eliminates the need for per-object tracking like what we did with
     FrameTracker -> orbTracker.
     */
    private void homogBasedTracker(long timestamp, Bitmap frame) {
        //1. Get the current frame transformation relative to reference frame.

        RefFrame referenceFrame = RefFrame.getInstance();

        long start = SystemClock.uptimeMillis();

        Mat refMat = referenceFrame.refMat;
        Mat currMat = new Mat();

        Utils.bitmapToMat(frame, currMat);
        Imgproc.cvtColor(currMat, currMat, Imgproc.COLOR_RGB2GRAY);

        MatOfPoint refPoints = referenceFrame.refPoints;
        MatOfPoint currFeatures = new MatOfPoint();

        Imgproc.goodFeaturesToTrack(currMat,currFeatures,50,0.1,10);

        MatOfPoint2f refPoints2F = new MatOfPoint2f(refPoints.toArray());
        MatOfPoint2f currPoints2F = new MatOfPoint2f();
        MatOfByte status = new MatOfByte();
        Video.calcOpticalFlowPyrLK(refMat,currMat,refPoints2F,currPoints2F, status, new MatOfFloat());

        //2. Updating the locations based on the tracking information
//        reference.second.location = location;
//        reference.second.lastUpdate = SystemClock.uptimeMillis();;

        Mat Homog = Calib3d.findHomography(refPoints2F, currPoints2F);

        final LinkedList<TrackedRecognition> copyList =
                new LinkedList<>(trackedObjects); // or cvTrackedObjects ..?

        trackedObjects.clear();

        for (TrackedRecognition recognition: copyList) {

            RectF location = recognition.location;

            // For every tracked object, update accordingly, i.e. warp previous loc according
            // to the computed homography.
            Mat ref_location = new Mat(4,1, CvType.CV_32FC2);
            Mat cur_location = new Mat(4,1,CvType.CV_32FC2);

            ref_location.put(0, 0, new double[] {location.left,location.top});
            ref_location.put(1, 0, new double[] {location.right,location.top});
            ref_location.put(2, 0, new double[] {location.right,location.bottom});
            ref_location.put(3, 0, new double[] {location.left,location.bottom});

            Core.perspectiveTransform(ref_location,cur_location,Homog);

            List<Point> points = new ArrayList<>();
            RectF newLocation = new RectF();

            for (int i=0; i < cur_location.rows(); i++) {
                Point point = new Point();
                point.set(cur_location.get(i,0));
                points.add(point);
            }

            float[] xValues = {(float) points.get(0).x,
                    (float) points.get(1).x,
                    (float) points.get(2).x,
                    (float) points.get(3).x};
            float[] yValues = {(float) points.get(0).y,
                    (float) points.get(1).y,
                    (float) points.get(2).y,
                    (float) points.get(3).y};
            Arrays.sort(xValues);
            Arrays.sort(yValues);
            newLocation.set(xValues[0], yValues[0], xValues[3], yValues[3]);

            try {
                recognition.location = newLocation;
                recognition.sensitivity = objectReferenceList.isSensitive(recognition.title);
                //if (recognition.sensitivity) {
                //    recognition.pathLocation = getContourMatch(currMat, newLocation);
                //}
                recognition.lastUpdate = SystemClock.uptimeMillis();
                trackedObjects.add(recognition);
            } catch (Exception e) {
                e.printStackTrace();
            }

            setFrame(frame);

        }

    }

    private void orbTracker(long timestamp,
                          List<Pair<Mat, TrackedRecognition>> references,
                          Bitmap frame){

    final ORB featureDetector = ORB.create();
    final DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

    Mat qryDescriptors = new Mat();
    MatOfKeyPoint qryKeypoints = new MatOfKeyPoint();


    Mat qryImage = new Mat();
    Utils.bitmapToMat(frame, qryImage);

    featureDetector.detect(qryImage, qryKeypoints);
    featureDetector.compute(qryImage, qryKeypoints, qryDescriptors);

    for (Pair<Mat, TrackedRecognition> reference: references){

      final String objectName = reference.second.title;

      ArrayList<Point> points = new ArrayList<>();
      List<MatOfPoint> mScenePoints = new ArrayList<>();

      Mat refDescriptors = new Mat();
      MatOfKeyPoint refKeypoints = new MatOfKeyPoint();
      final Mat refImage = reference.first;
      MatOfDMatch matches = new MatOfDMatch();

      featureDetector.detect(refImage, refKeypoints);
      featureDetector.compute(refImage, refKeypoints, refDescriptors);

      try{
        matcher.match(refDescriptors, qryDescriptors, matches);
        //match(refDescriptors, qryDescriptors, matches);

        long time = System.currentTimeMillis();

        //Using regular matching
        List<DMatch> matchesList = matches.toList();

        Double max_dist = 0.0;
        Double min_dist = 100.0;

        for (int i = 0; i < matchesList.size(); i++) {
          Double dist = (double) matchesList.get(i).distance;
          if (dist < min_dist)
            min_dist = dist;
          if (dist > max_dist)
            max_dist = dist;
        }

        // ratio test
        LinkedList<DMatch> good_matches = new LinkedList<>();
        for (int i = 0; i < matchesList.size(); i++) {
          if (matchesList.get(i).distance <= (1.5 * min_dist))
            good_matches.addLast(matchesList.get(i));
        }
/*            for (Iterator<MatOfDMatch> iterator = matches.iterator(); iterator.hasNext();) {
                MatOfDMatch matOfDMatch = iterator.next();
                if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.75) {
                    good_matches.add(matOfDMatch.toArray()[0]);
                }
            }*/

        long time1 = System.currentTimeMillis();

        if (good_matches.size() > minMatchCount){

          /** get keypoint coordinates of good matches to find homography and remove outliers
           * using ransac */
          /** Also, always remember that this is already a transformation process. */

          List<Point> refPoints = new ArrayList<>();
          List<Point> mPoints = new ArrayList<>();
          for(int i = 0; i<good_matches.size(); i++){
            refPoints.add(refKeypoints.toList().get(good_matches.get(i).queryIdx).pt);
            mPoints.add(qryKeypoints.toList().get(good_matches.get(i).trainIdx).pt);
          }
          // convertion of data types - there is maybe a more beautiful way
          Mat outputMask = new Mat();
          MatOfPoint2f rPtsMat = new MatOfPoint2f();
          rPtsMat.fromList(refPoints);
          MatOfPoint2f mPtsMat = new MatOfPoint2f();
          mPtsMat.fromList(mPoints);

          Mat obj_corners = new Mat(4,1, CvType.CV_32FC2);
          Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

          obj_corners.put(0, 0, new double[] {0,0});
          obj_corners.put(1, 0, new double[] {refImage.width()-1,0});
          obj_corners.put(2, 0, new double[] {refImage.width()-1,refImage.height()-1});
          obj_corners.put(3, 0, new double[] {0,refImage.height()-1});

          // Find homography - here just used to perform match filtering with RANSAC, but could be used to e.g. stitch images
          // the smaller the allowed reprojection error (here 15), the more matches are filtered
          Mat Homog = Calib3d.findHomography(rPtsMat, mPtsMat, Calib3d.RANSAC, 15, outputMask, 2000, 0.995);

          Core.perspectiveTransform(obj_corners,scene_corners,Homog);

          MatOfPoint sceneCorners = new MatOfPoint();
          for (int i=0; i < scene_corners.rows(); i++) {
            Point point = new Point();
            point.set(scene_corners.get(i,0));
            points.add(point);
          }
          sceneCorners.fromList(points);
          mScenePoints.add(sceneCorners);

          if (Imgproc.contourArea(mScenePoints.get(0)) > (minMatchCount*minMatchCount)) {
/*            logger.i("Time to Match: " + Long.toString((time1 - time))
                    + ", Number of matches: " + good_matches.size()
                    + " (" + Integer.toString(minMatchCount) + ")"
                    + ", Time to transform: " + Long.toString((System.currentTimeMillis() - time1)));*/
          } else {
            // Transformation is too small or skewed, object probably not in view, or matching
            // error.
/*            logger.i( "Time to Match: " + Long.toString((time1 - time))
                    + ", Object probably not in view even with " + good_matches.size()
                    + " (" + Integer.toString(minMatchCount) + ") matches.");*/

            return;
          }
          //result = "Enough matches.";
        } else {
/*          logger.i( "Time to Match: " + Long.toString((System.currentTimeMillis() - time))
                  + ", Not Enough Matches (" + good_matches.size()
                  + "/" + Integer.toString(minMatchCount) + ")");*/
          //result = "Not enough matches.";
          return;
        }

      } catch (Exception e) {
        e.printStackTrace();
/*
        logger.d("Cannot process.");
*/
        return;
      }

      /**
       * Using RectF to draw a fixed rectangle bounding box.
       */
      float[] xValues = {(float) points.get(0).x,
              (float) points.get(1).x,
              (float) points.get(2).x,
              (float) points.get(3).x};
      float[] yValues = {(float) points.get(0).y,
              (float) points.get(1).y,
              (float) points.get(2).y,
              (float) points.get(3).y};
      Arrays.sort(xValues);
      Arrays.sort(yValues);
      RectF location = new RectF(xValues[0], yValues[0], xValues[3], yValues[3]);

      reference.second.location = location;
      reference.second.lastUpdate = SystemClock.uptimeMillis();;

      final LinkedList<TrackedRecognition> copyList =
              new LinkedList<TrackedRecognition>(trackedObjects);

      trackedObjects.clear();
      for (TrackedRecognition recognition: copyList) {
        if (recognition.title == objectName){
          RectF previousLocation = recognition.location;
          double prevX = previousLocation.centerX();
          double prevY = previousLocation.centerY();
          double locX = location.centerX();
          double locY = location.centerY();
          double diff = Math.sqrt((prevX - locX)*(prevX - locX) + (prevY - locY)*(prevY - locY));

          if (diff < TRACKING_DIFFERENCE) {
            logger.i("Diff: %f", diff);

            // If same name and is less than 50 pixels away, update the existing tracked object.
            recognition.detectionConfidence = reference.second.detectionConfidence;
            recognition.lastUpdate = SystemClock.uptimeMillis();;
            //recognition.location = location;
          }
          trackedObjects.add(recognition);

        } else {
          trackedObjects.add(reference.second); // New/updated detections are added.
        }
      }

    }

  }

    public void setFrame(Bitmap bitmap){
        RefFrame referenceFrame = RefFrame.getInstance();

        referenceFrame.refFrame = bitmap;

        Mat refMat = new Mat();
        MatOfPoint refFeatures = new MatOfPoint();

        Utils.bitmapToMat(bitmap, refMat);
        Imgproc.cvtColor(refMat, refMat, Imgproc.COLOR_RGB2GRAY);

        Imgproc.goodFeaturesToTrack(refMat,refFeatures,50,0.1,10);

        referenceFrame.refMat = refMat;
        referenceFrame.refPoints = refFeatures;
    }

    private Mat histogram(Bitmap bitmap){

        Mat img = new Mat();
        Utils.bitmapToMat(bitmap, img);
        Imgproc.cvtColor(img, img, Imgproc.COLOR_RGB2GRAY);

        Vector<Mat> bgr_planes = new Vector<Mat>();
        Core.split(img, bgr_planes);

        MatOfInt histSize = new MatOfInt(256);

        final MatOfFloat histRange = new MatOfFloat(0f, 256f);

        boolean accumulate = false;
        Mat b_hist = new  Mat();

        Imgproc.calcHist(bgr_planes, new MatOfInt(0),new Mat(), b_hist, histSize, histRange, accumulate);
        return b_hist;

    }

    private Mat calcOpticalFlow(Bitmap prev, Bitmap next){

        long start = SystemClock.uptimeMillis();

        Mat prevMat = new Mat();
        Utils.bitmapToMat(prev, prevMat);
        Imgproc.cvtColor(prevMat, prevMat, Imgproc.COLOR_RGB2GRAY);

        Mat nextMat = new Mat();
        Utils.bitmapToMat(next, nextMat);
        Imgproc.cvtColor(nextMat, nextMat, Imgproc.COLOR_RGB2GRAY);

        Mat flow = new Mat(prevMat.size(), CvType.CV_8UC1);

        Video.calcOpticalFlowFarneback(prevMat, nextMat, flow,0.5,1, 1, 1, 5,
                1.1,1);

        logger.i("Optical Flow time: %d", SystemClock.uptimeMillis() - start);

        return flow;

    }

    private Path getContourMatch(Mat frame, RectF location){

        Path contourMatchPath = new Path();

//        Mat gray = new Mat();
//        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        // python: edges = cv2.Canny(gray, 10, 100)
        Mat edges = new Mat();
        Imgproc.Canny(frame, edges, 50, 500,3,false);

        Mat mask = new Mat(frame.height(), frame.width(), CvType.CV_8UC1);
        Mat mask2 = new Mat();
        Imgproc.rectangle(mask,
                new Point(0.75f*location.left,0.75f*location.top),
                new Point(1.24f*location.right,1.25f*location.bottom),
                new Scalar(255));

        logger.d("Mask width = %d, height = %d, channel = %d.", mask.width(), mask.height(), mask.channels());
        logger.d("Location width = %.3f, height = %.3f", location.width(), location.height());
        logger.d("Frame width = %d, height = %d, channel = %d.",frame.width(), frame.height(), frame.channels());

        Mat masked = new Mat();
        Core.bitwise_or(frame,mask,masked);

        // python: contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(masked, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        logger.i("ContourMatching: detected contours %d",contours.size());

        double minContourLength = Math.max(location.width(), location.height());

        MatOfPoint goodContour = new MatOfPoint();

        double maxContourLength = 0;

        for (MatOfPoint contour : contours) {
            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            double contourLength = Imgproc.arcLength(curve, true);
            //final MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());

            //double contourArea = Imgproc.contourArea(contour);
            // len(contour) --> contour.toList().size() //&& contourLength<=maxContourLength)
            if (contourLength >= minContourLength) {
                if (contourLength > maxContourLength) {
                    maxContourLength = contourLength;
                    goodContour = contour;
                }
            }

        } //getting long enough contours

        List<Point> contourPoints = goodContour.toList();

        if (!contourPoints.isEmpty()) { // We got a contour that is possibly the object.

            boolean pointInitialized = false;

            for (Point point : contourPoints) {
                logger.i("%.3f, %.3f", point.x, point.y);
                if (!pointInitialized) {
                    contourMatchPath.moveTo((float) point.x, (float) point.y);
                    pointInitialized = true;
                } else contourMatchPath.lineTo((float) point.x, (float) point.y);
            }

            contourMatchPath.close();

            logger.i("ContourMatching: Detected a potentially matched contour (path)!"); //%s", contourMatchPath.);
        }

        return contourMatchPath;

    }

    private void scaleRectF(RectF rect, float factor){
        float diffHorizontal = (rect.right-rect.left) * (factor-1f);
        float diffVertical = (rect.bottom-rect.top) * (factor-1f);

        rect.top -= diffVertical/2f;
        rect.bottom += diffVertical/2f;

        rect.left -= diffHorizontal/2f;
        rect.right += diffHorizontal/2f;
    }

    private void processResults(
            final long timestamp, final List<Recognition> results, final byte[] originalFrame) {

    final List<Pair<Float, Recognition>> rectsToTrack = new LinkedList<Pair<Float, Recognition>>();

    screenRects.clear();
    final Matrix rgbFrameToScreen = new Matrix(getFrameToCanvasMatrix());

    for (final Recognition result : results) {
      if (result.getLocation() == null) {
        continue;
      }

      final RectF detectionFrameRect = new RectF(result.getLocation());

      final RectF detectionScreenRect = new RectF();
      rgbFrameToScreen.mapRect(detectionScreenRect, detectionFrameRect);

      logger.v(
          "Result! Frame: " + result.getLocation() + " mapped to screen:" + detectionScreenRect);

      screenRects.add(new Pair<Float, RectF>(result.getConfidence(), detectionScreenRect));

      if (detectionFrameRect.width() < MIN_SIZE || detectionFrameRect.height() < MIN_SIZE) {
        logger.w("Degenerate rectangle! " + detectionFrameRect);
        continue;
      }

      rectsToTrack.add(new Pair<Float, Recognition>(result.getConfidence(), result));
    }

    if (rectsToTrack.isEmpty()) {
      logger.v("Nothing to track, aborting.");
      return;
    }

    if (objectTracker == null) {
      trackedObjects.clear();
      for (final Pair<Float, Recognition> potential : rectsToTrack) {

        final TrackedRecognition trackedRecognition = new TrackedRecognition();

        // Separate handling for results from Marker detection
        if (potential.second.getId() == "Marker") {
          trackedRecognition.detectionConfidence = potential.first;
          trackedRecognition.location = new RectF(potential.second.getLocation());
          trackedRecognition.trackedObject = null;
          trackedRecognition.title = potential.second.getId() + potential.second.getTitle();
          trackedRecognition.color = COLORS[trackedObjects.size()];
          //trackedRecognition.lastUpdate = SystemClock.uptimeMillis();
          //trackedRecognition.sensitivity = false;

          trackedObjects.add(trackedRecognition);
          continue;
        }

        trackedRecognition.detectionConfidence = potential.first;
        trackedRecognition.location = new RectF(potential.second.getLocation());
        trackedRecognition.trackedObject = null;
        trackedRecognition.title = potential.second.getTitle();
        trackedRecognition.color = COLORS[trackedObjects.size()];
        trackedRecognition.lastUpdate = SystemClock.uptimeMillis();
        trackedRecognition.sensitivity = objectReferenceList.isSensitive(trackedRecognition.title);

        trackedObjects.add(trackedRecognition);

        if (trackedObjects.size() >= COLORS.length) {
          break;
        }
      }

      return;
    }

    logger.i("%d rects to track", rectsToTrack.size());
    for (final Pair<Float, Recognition> potential : rectsToTrack) {
      handleDetection(originalFrame, timestamp, potential);
    }
    }

    private void handleDetection(
            final byte[] frameCopy, final long timestamp, final Pair<Float, Recognition> potential) {
    final ObjectTracker.TrackedObject potentialObject =
        objectTracker.trackObject(potential.second.getLocation(), timestamp, frameCopy);

    final float potentialCorrelation = potentialObject.getCurrentCorrelation();
    logger.v(
        "Tracked object went from %s to %s with correlation %.2f",
        potential.second, potentialObject.getTrackedPositionInPreviewFrame(), potentialCorrelation);

    if (potentialCorrelation < MARGINAL_CORRELATION) {
      logger.v("Correlation too low to begin tracking %s.", potentialObject);
      potentialObject.stopTracking();
      return;
    }

    final List<TrackedRecognition> removeList = new LinkedList<TrackedRecognition>();

    float maxIntersect = 0.0f;

    // This is the current tracked object whose color we will take. If left null we'll take the
    // first one from the color queue.
    TrackedRecognition recogToReplace = null;

    // Look for intersections that will be overridden by this object or an intersection that would
    // prevent this one from being placed.
    for (final TrackedRecognition trackedRecognition : trackedObjects) {
      final RectF a = trackedRecognition.trackedObject.getTrackedPositionInPreviewFrame();
      final RectF b = potentialObject.getTrackedPositionInPreviewFrame();
      final RectF intersection = new RectF();
      final boolean intersects = intersection.setIntersect(a, b);

      final float intersectArea = intersection.width() * intersection.height();
      final float totalArea = a.width() * a.height() + b.width() * b.height() - intersectArea;
      final float intersectOverUnion = intersectArea / totalArea;

      // If there is an intersection with this currently tracked box above the maximum overlap
      // percentage allowed, either the new recognition needs to be dismissed or the old
      // recognition needs to be removed and possibly replaced with the new one.
      if (intersects && intersectOverUnion > MAX_OVERLAP) {
        if (potential.first < trackedRecognition.detectionConfidence
            && trackedRecognition.trackedObject.getCurrentCorrelation() > MARGINAL_CORRELATION) {
          // If track for the existing object is still going strong and the detection score was
          // good, reject this new object.
          potentialObject.stopTracking();
          return;
        } else {
          removeList.add(trackedRecognition);

          // Let the previously tracked object with max intersection amount donate its color to
          // the new object.
          if (intersectOverUnion > maxIntersect) {
            maxIntersect = intersectOverUnion;
            recogToReplace = trackedRecognition;
          }
        }
      }
    }

    // If we're already tracking the max object and no intersections were found to bump off,
    // pick the worst current tracked object to remove, if it's also worse than this candidate
    // object.
    if (availableColors.isEmpty() && removeList.isEmpty()) {
      for (final TrackedRecognition candidate : trackedObjects) {
        if (candidate.detectionConfidence < potential.first) {
          if (recogToReplace == null
              || candidate.detectionConfidence < recogToReplace.detectionConfidence) {
            // Save it so that we use this color for the new object.
            recogToReplace = candidate;
          }
        }
      }
      if (recogToReplace != null) {
        logger.v("Found non-intersecting object to remove.");
        removeList.add(recogToReplace);
      } else {
        logger.v("No non-intersecting object found to remove");
      }
    }

    // Remove everything that got intersected.
    for (final TrackedRecognition trackedRecognition : removeList) {
      logger.v(
          "Removing tracked object %s with detection confidence %.2f, correlation %.2f",
          trackedRecognition.trackedObject,
          trackedRecognition.detectionConfidence,
          trackedRecognition.trackedObject.getCurrentCorrelation());
      trackedRecognition.trackedObject.stopTracking();
      trackedObjects.remove(trackedRecognition);
      if (trackedRecognition != recogToReplace) {
        availableColors.add(trackedRecognition.color);
      }
    }

    if (recogToReplace == null && availableColors.isEmpty()) {
      logger.e("No room to track this object, aborting.");
      potentialObject.stopTracking();
      return;
    }

    // Finally safe to say we can track this object.
    logger.v(
        "Tracking object %s (%s) with detection confidence %.2f at position %s",
        potentialObject,
        potential.second.getTitle(),
        potential.first,
        potential.second.getLocation());
    final TrackedRecognition trackedRecognition = new TrackedRecognition();
    trackedRecognition.detectionConfidence = potential.first;
    trackedRecognition.trackedObject = potentialObject;
    trackedRecognition.title = potential.second.getTitle();
    trackedRecognition.lastUpdate = SystemClock.uptimeMillis();;

    // Use the color from a replaced object before taking one from the color queue.
    trackedRecognition.color =
        recogToReplace != null ? recogToReplace.color : availableColors.poll();
    trackedObjects.add(trackedRecognition);
    }

}
