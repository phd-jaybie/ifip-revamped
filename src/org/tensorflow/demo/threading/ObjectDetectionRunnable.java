package org.tensorflow.demo.threading;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;

import org.tensorflow.demo.Classifier;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.phd.detector.cv.CvDetector;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by deg032 on 22/5/18.
 */

public class ObjectDetectionRunnable implements Runnable {

    private static final Logger LOGGER = new Logger();

    // Constants for indicating the state of the decode
    static final int DETECTION_STATE_FAILED = -1;
    static final int DETECTION_STATE_STARTED = 0;
    static final int DETECTION_STATE_COMPLETED = 1;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

    // Defines a field that contains the calling object of type PhotoTask.
    final TaskRunnableObjectDetectionMethods mProcessTask;

    private static Classifier mDetector; //for TF detection
    private static Bitmap mInputBitmap;
    private static long mCurrentFrame;

    /**
     *
     * An interface that defines methods that PhotoTask implements. An instance of
     * PhotoTask passes itself to an PhotoDownloadRunnable instance through the
     * PhotoDownloadRunnable constructor, after which the two instances can access each other's
     * variables.
     */
    interface TaskRunnableObjectDetectionMethods {

        /**
         * Sets the Thread that this instance is running on
         * @param currentThread the current Thread
         */
        void setObjectDetectionThread(Thread currentThread);

        /**
         * Defines the actions for each state of the PhotoTask instance.
         * @param state The current state of the task
         */
        void handleObjectDetectionState(int state);

        Bitmap getInputBitmap();

        Classifier getDetector();

        long getCurrentFrame();

        void setTFResults(List<Classifier.Recognition> results);

    }

    ObjectDetectionRunnable(TaskRunnableObjectDetectionMethods processTask){//, Bitmap inputBitmap, Classifier detector) {
        mProcessTask = processTask;
        //mInputBitmap = inputBitmap;
        //mDetector = detector;
    }

    @Override
    public void run() {

        final long startTime = SystemClock.uptimeMillis();

        List<Classifier.Recognition> results = new ArrayList<>();

        /*
        croppedBitmap

        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
        final Canvas canvas = new Canvas(cropCopyBitmap);
        final Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);
        */

        /*
         * Stores the current Thread in the the PhotoTask instance, so that the instance
         * can interrupt the Thread.
         */
        mProcessTask.setObjectDetectionThread(Thread.currentThread());

        // Moves the current Thread into the background
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND);

        mProcessTask.handleObjectDetectionState(DETECTION_STATE_STARTED);

        mInputBitmap = mProcessTask.getInputBitmap();
        mDetector = mProcessTask.getDetector();
        mCurrentFrame = mProcessTask.getCurrentFrame();

        if (mInputBitmap==null || mDetector==null) {
            LOGGER.d("Detection Failed.");
            mProcessTask.handleObjectDetectionState(DETECTION_STATE_FAILED);
            return;
        }

        try {
            if (Thread.interrupted()) {
                throw new InterruptedException();
            }

            results = mDetector.recognizeImage(mInputBitmap);
            mProcessTask.handleObjectDetectionState(DETECTION_STATE_COMPLETED);

        } catch (Exception e){

            e.printStackTrace();

        } finally {

            if (results == null) {
                LOGGER.d("Detection Failed.");
                mProcessTask.handleObjectDetectionState(DETECTION_STATE_FAILED);
                return;
            }

            LOGGER.d("Detection Completed.");

            final long detectionTime = SystemClock.uptimeMillis() - startTime;
            mProcessTask.setTFResults(processTFResults(results));

            LOGGER.i("%d, Detection Time: %d" , mCurrentFrame , detectionTime);

            // Sets the reference to the current Thread to null, releasing its storage
            mProcessTask.setObjectDetectionThread(null);

            // Clears the Thread's interrupt flag
            Thread.interrupted();

        }

    }

    static List<Classifier.Recognition> processTFResults(List<Classifier.Recognition> results){

        final List<Classifier.Recognition> appResults =
                new LinkedList<>(); // collection of results per app

        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

        //transformation
        for (final Classifier.Recognition dResult : results) {
            final RectF location = dResult.getLocation();
            if (location != null && dResult.getConfidence() >= minimumConfidence) {

                /*inputToCropTransform.mapRect(location);
                cropCanvas.drawRect(location, paint);

                if (Arrays.asList(secretObjects).contains(dResult.getTitle())){
                    localSecrecyHit = 1;
                    //continue; //Don't overlay if object is secret.
                } else if (objectsOfInterest.contains(dResult.getTitle())){ // case 3
                    localHit = 1;
                }

                cropToFrameTransform.mapRect(location);
                */

                dResult.setLocation(location);
                appResults.add(dResult);

                //localHit = 1;

            }
        }

        return appResults;
    }

    static Classifier.Recognition processCVResults(CvDetector.Recognition result){

        result.setTitle("CV_DETECTOR");

        Classifier.Recognition cvDetection = new Classifier.
                Recognition("SIFT/ORB", result.getTitle(),
                MINIMUM_CONFIDENCE_TF_OD_API, result.getLocation().second);

        return cvDetection;
    }
}
