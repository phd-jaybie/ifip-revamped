package org.tensorflow.ifip.threading;

import android.graphics.Bitmap;

import org.tensorflow.ifip.phd.detector.cv.CvDetector;

/**
 * Created by deg032 on 22/5/18.
 */

public class FrameTrackingRunnable implements Runnable {

    // Constants for indicating the state of the decode
    static final int TRACKING_STATE_FAILED = -1;
    static final int TRACKING_STATE_STARTED = 0;
    static final int TRACKING_STATE_COMPLETED = 1;

    // Defines a field that contains the calling object of type PhotoTask.
    final TaskRunnableFrameTrackingMethods mProcessTask;

    private Bitmap mInputBitmap;

    /**
     *
     * An interface that defines methods that PhotoTask implements. An instance of
     * PhotoTask passes itself to an PhotoDownloadRunnable instance through the
     * PhotoDownloadRunnable constructor, after which the two instances can access each other's
     * variables.
     */
    interface TaskRunnableFrameTrackingMethods {

        /**
         * Sets the Thread that this instance is running on
         * @param currentThread the current Thread
         */
        void setFrameTrackingThread(Thread currentThread);

        /**
         * Defines the actions for each state of the PhotoTask instance.
         * @param state The current state of the task
         */
        void handleFrameTrackingState(int state);

        Bitmap getInputBitmap();

        void setCVResults(CvDetector.Recognition result);

    }

    FrameTrackingRunnable(TaskRunnableFrameTrackingMethods processTask){//},  Bitmap inputBitmap) {
        mProcessTask = processTask;
        //mInputBitmap = inputBitmap;
    }

    @Override
    public void run() {
        // DO whatever
    }
}
