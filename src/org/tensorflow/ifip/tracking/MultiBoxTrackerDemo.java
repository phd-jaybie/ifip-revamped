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

package org.tensorflow.ifip.tracking;

import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Cap;
import android.graphics.Paint.Join;
import android.graphics.Paint.Style;
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
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.tensorflow.ifip.Classifier.Recognition;
import org.tensorflow.ifip.env.BorderedText;
import org.tensorflow.ifip.env.ImageUtils;
import org.tensorflow.ifip.env.Logger;
import org.tensorflow.ifip.initializer.ObjectReferenceList;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Vector;

import static org.tensorflow.ifip.MrCameraActivity.MIN_MATCH_COUNT;

/**
 * A tracker wrapping ObjectTracker that also handles non-max suppression and matching existing
 * objects to new detections.
 */

public class MultiBoxTrackerDemo {

  int minMatchCount = Math.round(MIN_MATCH_COUNT*10/30);

  private final Logger logger = new Logger();

  private static final float TEXT_SIZE_DIP = 18;

  private static final int TRACKING_TIMEOUT = 1000;

  private static final int TRACKING_DIFFERENCE = 100;

  // Maximum percentage of a box that can be overlapped by another box at detection time. Otherwise
  // the lower scored box (new or old) will be removed.
  private static final float MAX_OVERLAP = 0.2f;

  private static final float MIN_SIZE = 16.0f;

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

  private Bitmap prevFrame;

  public void setFrame(Bitmap bitmap){
    prevFrame = bitmap;
  }

  private static class TrackedRecognition {
    ObjectTracker.TrackedObject trackedObject;
    RectF location;
    float detectionConfidence;
    int color;
    String title;
    long lastUpdate;
    int coverColor;
    boolean sensitivity = false;
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

  public MultiBoxTrackerDemo(final Context context) {
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

  }

  private Matrix getFrameToCanvasMatrix() {
    return frameToCanvasMatrix;
  }

  public synchronized void drawDebug(final Canvas canvas) {
    final Paint textPaint = new Paint();
    textPaint.setColor(Color.WHITE);
    textPaint.setTextSize(60.0f);

    final Paint boxPaint = new Paint();
    boxPaint.setColor(Color.RED);
    boxPaint.setAlpha(200);
    boxPaint.setStyle(Style.STROKE);

    for (final Pair<Float, RectF> detection : screenRects) {
      final RectF rect = detection.second;
      canvas.drawRect(rect, boxPaint);
      canvas.drawText("" + detection.first, rect.left, rect.top, textPaint);
      borderedText.drawText(canvas, rect.centerX(), rect.centerY(), "" + detection.first);
    }

    if (objectTracker == null) {
      logger.i("DrawDebug: Object Tracker is null.");
      return;
    }

    // Draw correlations.
    for (final TrackedRecognition recognition : trackedObjects) {
      final ObjectTracker.TrackedObject trackedObject = recognition.trackedObject;

      final RectF trackedPos = trackedObject.getTrackedPositionInPreviewFrame();

      if (getFrameToCanvasMatrix().mapRect(trackedPos)) {
        final String labelString = String.format("%.2f", trackedObject.getCurrentCorrelation());
        borderedText.drawText(canvas, trackedPos.right, trackedPos.bottom, labelString);
      }
    }

    final Matrix matrix = getFrameToCanvasMatrix();
    objectTracker.drawDebug(canvas, matrix);
  }

  public synchronized void trackResults(
      final List<Recognition> results, final byte[] frame, final long timestamp) {
    logger.i("Processing %d results from %d", results.size(), timestamp);
    processResults(timestamp, results, frame);
  }

  public synchronized void draw(final Canvas canvas) {
    final boolean rotated = sensorOrientation % 180 == 90;
    final float multiplier =
        Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                 canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
    frameToCanvasMatrix =
        ImageUtils.getTransformationMatrix(
            frameWidth,
            frameHeight,
            (int) (multiplier * (rotated ? frameHeight : frameWidth)),
            (int) (multiplier * (rotated ? frameWidth : frameHeight)),
            sensorOrientation,
            false);

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

        // expanded the destination Rect to prevent leaks
        final RectF secretPos = new RectF(
                0.9f*trackedPos.left,
                1.1f*trackedPos.top,
                1.1f*trackedPos.right,
                0.9f*trackedPos.bottom);

        /*Bitmap bitmap = BitmapFactory.decodeResource(res, R.drawable.csiro);
        canvas.drawBitmap(bitmap,
                new Rect(0,0,bitmap.getWidth(),bitmap.getHeight()),
                trackedPos,
                boxPaint);*/

        Paint fillPaint = new Paint();
        fillPaint.setStyle(Style.FILL);
        fillPaint.setColor(recognition.coverColor);

        //float cornerSize = Math.min(secretPos.width(), secretPos.height()) / 16.0f;
        canvas.drawRect(trackedPos, fillPaint);    // fill

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

  public void setInitialize(boolean set){
    initialized = set;
  }

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

  public synchronized void FrameTracker(
          final int w,
          final int h,
          final int sensorOrientation,
          final Bitmap frame,
          final long timestamp) {

    if (trackedObjects.isEmpty()) {
      logger.i("Nothing new to track.");
      return;
    }

    refreshTrackedObjects(frame);

    if (!initialized) {
      cvTrackedObjects.clear();

      logger.i("%d, Initializing CVTracker: %dx%d", timestamp, w, h);

      for (final TrackedRecognition recognition: trackedObjects) {

        CvTrackedRecognition cvTrackedRecognition = new CvTrackedRecognition();
        Mat currentFrame = new Mat();

        RectF location = recognition.location;
        Rect roundedLocation = new Rect();
        location.round(roundedLocation);

/*
        logger.i("%d, FrameW: %d, FrameH: %d", timestamp, frame.getWidth(), frame.getHeight());

        logger.i("%d, Object: %s, RectFL: %f, RectFT: %f, RectFR: %f, RectFB: %f",
                timestamp, recognition.title, location.left, location.top, location.right, location.bottom);

        logger.i("%d, Object: %s, RectFX: %d, RectFY: %d, RectFW: %d, RectFH: %d",
                timestamp, recognition.title,
                roundedLocation.left,
                roundedLocation.top,
                roundedLocation.right - roundedLocation.left,
                roundedLocation.bottom - roundedLocation.top
        );
*/

        final Bitmap refImage = Bitmap.createBitmap(frame,
                Math.abs(roundedLocation.left),
                Math.abs(roundedLocation.top),
                Math.min((roundedLocation.right - roundedLocation.left),(frame.getWidth()-Math.abs(roundedLocation.left))),
                Math.min((roundedLocation.bottom - roundedLocation.top),(frame.getHeight()-Math.abs(roundedLocation.top)))
        );

        Utils.bitmapToMat(refImage, currentFrame);
        logger.i("%d, MatW: %d, MatH: %d",
                timestamp,
                currentFrame.cols(),
                currentFrame.rows());
        final int coverColor = refImage.getPixel(0, 0);

        cvTrackedRecognition.trackedRecognition = recognition;
        cvTrackedRecognition.location = recognition.location;
        cvTrackedRecognition.color = recognition.color;
        cvTrackedRecognition.detectionConfidence = recognition.detectionConfidence;
        cvTrackedRecognition.title = recognition.title;
        cvTrackedRecognition.RefImageMat = currentFrame;
        cvTrackedRecognition.trackedRecognition.coverColor = coverColor;


        cvTrackedObjects.add(cvTrackedRecognition);
      }

      frameWidth = w;
      frameHeight = h;
      this.sensorOrientation = sensorOrientation;
      initialized = true;
      return;
    }

    if (cvTrackedObjects.isEmpty()){
      logger.i("Nothing tracked.");
      return;
    }

    //Mat H1 = histogram(prevFrame);
    //Mat H2 = histogram(frame);
    //double frameSimilarity = Imgproc.compareHist(H1, H2, Imgproc.CV_COMP_BHATTACHARYYA);

    //logger.i("%d: Frame Correlation: %f", timestamp,frameSimilarity);

    //Mat flow = calcOpticalFlow(prevFrame, frame);

/*    Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
    try {
      Utils.matToBitmap(flow, bmp);
    } catch (CvException e) {
      logger.d("CV Exception",e.getMessage());
    }

    FileOutputStream out = null;
    try {
      out = new FileOutputStream("flow.png");
      bmp.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
      // PNG is a lossless format, the compression factor (100) is ignored
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      try {
        if (out != null) {
          out.close();
        }
      } catch (IOException e) {
        e.printStackTrace();
      }
    }*/

    //trackedObjects.clear(); // Does not have to be cleared everytime.

    long start = System.currentTimeMillis();

    List<Pair<Mat, TrackedRecognition>> refImageMats = new ArrayList<>();

    for (final CvTrackedRecognition cvTrackedRecognition: cvTrackedObjects){
      logger.i("Tracking object: " + cvTrackedRecognition.title);

      refImageMats.add(new Pair<>(cvTrackedRecognition.RefImageMat,cvTrackedRecognition.trackedRecognition));
    }

    orbTracker(timestamp,refImageMats, frame);

/*    if (!results.isEmpty()) {
      for (final TrackedRecognition result: results){
        trackedObjects.add(result);
      }
    }*/

    long end = System.currentTimeMillis() - start;
    logger.i("CV Frame tracking time: %d", end);

    //objectTracker.nextFrame(frame, null, timestamp, null, true);

    // Clean up any objects not worth tracking any more.
/*
    final LinkedList<TrackedRecognition> copyList =
            new LinkedList<TrackedRecognition>(trackedObjects);
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
*/
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
