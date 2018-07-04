package org.tensorflow.ifip.threading;

import android.graphics.Bitmap;

import org.tensorflow.ifip.Classifier;
import org.tensorflow.ifip.phd.detector.cv.CvDetector;
import org.tensorflow.ifip.tracking.MultiBoxTracker;

import java.lang.ref.WeakReference;
import java.util.List;

/**
 * Created by deg032 on 22/5/18.
 */
public class ProcessTask implements
        FrameTrackingRunnable.TaskRunnableFrameTrackingMethods,
        ObjectDetectionRunnable.TaskRunnableObjectDetectionMethods {

    /*
     * Field containing the Thread this task is running on.
     */
    Thread mThreadThis;

    /*
     * Fields containing references to the two runnable objects that handle downloading and
     * decoding of the image.
     */
    private Runnable mObjectDetectionRunnable;
    private Runnable mFrameTrackingRunnable;

    // The Thread on which this task is currently running.
    private Thread mCurrentThread;

    // Delegates handling the current state of the task to the ProcessManager object
    void handleState(int state) {
        sProcessManager.handleState(this, state);
    }

    /*
     * An object that contains the ThreadPool singleton.
     */
    private static ProcessManager sProcessManager;
    private WeakReference<MultiBoxTracker> mTrackerWeakRef; // This is the current tracker from the UI.

    // Input Bitmap frame to be processed
    private Bitmap mInputFrame;
    private byte[] mLuminanceCopy;
    private long mCurrentFrame;

    private List<Classifier.Recognition> mResults;
    private Classifier mDetector; //for TF detection
    private CvDetector cvDetector; //for OpenCV detection

    /**
     * Creates a PhotoTask containing a download object and a decoder object.
     */
    ProcessTask() {
        // Create the runnables
        mObjectDetectionRunnable = new ObjectDetectionRunnable(this);
        mFrameTrackingRunnable = new FrameTrackingRunnable(this);//, mInputFrame);
        sProcessManager = ProcessManager.getInstance();
    }

    /*
 * Returns the Thread that this Task is running on. The method must first get a lock on a
 * static field, in this case the ThreadPool singleton. The lock is needed because the
 * Thread object reference is stored in the Thread object itself, and that object can be
 * changed by processes outside of this app.
 */
    public Thread getCurrentThread() {
        synchronized(sProcessManager) {
            return mCurrentThread;
        }
    }

    /*
     * Sets the identifier for the current Thread. This must be a synchronized operation; see the
     * notes for getCurrentThread()
     */
    public void setCurrentThread(Thread thread) {
        synchronized(sProcessManager) {
            mCurrentThread = thread;
        }
    }

    /**
     * Recycles an Process Task object before it's put back into the pool. One reason to do
     * this is to avoid memory leaks.
     **/
    void recycle() {

        // Deletes the weak reference to the imageView
        // Deletes the weak reference to the imageView
        if ( null != mTrackerWeakRef ) {
            mTrackerWeakRef.clear();
            mTrackerWeakRef = null;
        }

        mCurrentFrame = 0;

        // Releases references to the byte buffer and the BitMap
        if (mResults!=null) mResults.clear();
        mInputFrame = null;
        mLuminanceCopy = null;

    }

    /**
     * Initializes the Task
     *
     * @param processManager A ThreadPool object
     * @param tracker An MultiBox tracker from the Main.UI Thread that handles the rendering.
     * @param inputBitmap current inputBitmap to be processed.
     */
    void initializeDetector(
            ProcessManager processManager,
            MultiBoxTracker tracker,
            Bitmap inputBitmap,
            byte[] luminanceCopy,
            long currentFrame)
    {
        // Sets this object's ThreadPool field to be the input argument
        sProcessManager = processManager;

        // Gets the URL for the View
        mTrackerWeakRef = new WeakReference<>(tracker);

        mInputFrame = inputBitmap;

        mLuminanceCopy = luminanceCopy;
        mCurrentFrame = currentFrame;

    }

    void setTFDetector(Classifier detector){
        mDetector = detector;
    }

    void setCVDetector(CvDetector detector){
        cvDetector = detector;
    }

    @Override
    public void setFrameTrackingThread(Thread currentThread) { setCurrentThread(currentThread);
    }

    @Override
    public void handleFrameTrackingState(int state) {
        int outState;

        // Converts the decode state to the overall state.
        switch(state) {
            case FrameTrackingRunnable.TRACKING_STATE_COMPLETED:
                outState = ProcessManager.TRACKING_COMPLETE;
                break;
            //case FrameTrackingRunnable.TRACKING_STATE_FAILED:
            //    outState = ProcessManager.TRA;
            //    break;
            default:
                outState = ProcessManager.TRACKING_STARTED;
                break;
        }

        // Passes the state to the ThreadPool object.
        handleState(outState);
    }

    @Override
    public Bitmap getInputBitmap() {
        return mInputFrame;
    }

    public List<Classifier.Recognition> getResults(){
        return mResults;
    };

    public byte[] getLuminanceCopy(){
        return mLuminanceCopy;
    };

    @Override
    public long getCurrentFrame(){
        return mCurrentFrame;
    }

    @Override
    public void setTFResults(List<Classifier.Recognition> results){
        mResults = results;
    };

    @Override
    public void setCVResults(CvDetector.Recognition result){

        result.setTitle("CV_DETECTOR");

        Classifier.Recognition cvDetection = new Classifier.
                Recognition("SIFT/ORB", result.getTitle(),
                0.6f, result.getLocation().second);

        mResults.add(cvDetection);
    };

    @Override
    public void setObjectDetectionThread(Thread currentThread) {
        setCurrentThread(currentThread);
    }

    @Override
    public void handleObjectDetectionState(int state) {
        int outState;

        // Converts the decode state to the overall state.
        switch(state) {
            case ObjectDetectionRunnable.DETECTION_STATE_COMPLETED:
                outState = ProcessManager.DETECTION_COMPLETE;
                break;
            case ObjectDetectionRunnable.DETECTION_STATE_FAILED:
                outState = ProcessManager.DETECTION_FAILED;
                break;
            default:
                outState = ProcessManager.DETECTION_STARTED;
                break;
        }

        // Passes the state to the ThreadPool object.
        handleState(outState);
    };

    @Override
    public Classifier getDetector(){
        return mDetector;
    }

    public MultiBoxTracker getTracker(){
        if (mTrackerWeakRef!=null){
            return mTrackerWeakRef.get();
        }
        return null;
    }

    // Returns the instance that downloaded the image
    Runnable getObjectDetectionRunnable() {
        return mObjectDetectionRunnable;
    }

    // Returns the instance that decode the image
    Runnable getFrameTrackingRunnable() {
        return mFrameTrackingRunnable;
    }

}

