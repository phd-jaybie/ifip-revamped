package org.tensorflow.ifip.threading;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;

import org.tensorflow.ifip.Classifier;
import org.tensorflow.ifip.phd.detector.cv.CvDetector;
import org.tensorflow.ifip.simulator.AppRandomizer;
import org.tensorflow.ifip.tracking.MultiBoxTracker;

import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Created by deg032 on 22/5/18.
 */

public class ProcessManager{
    /*
    * Status indicators
    */
    public static final int DETECTION_FAILED = -1;
    public static final int DETECTION_STARTED = 1;
    public static final int DETECTION_COMPLETE = 2;
    public static final int TRACKING_STARTED = 3;
    public static final int TRACKING_COMPLETE = 4;

    // Sets the amount of time an idle thread will wait for a task before terminating
    private static final int KEEP_ALIVE_TIME = 1;

    // Sets the Time Unit to seconds
    private static final TimeUnit KEEP_ALIVE_TIME_UNIT;

    // Sets the initial threadpool size to 8
    private static final int CORE_POOL_SIZE = 8;

    // Sets the maximum threadpool size to 8
    private static final int MAXIMUM_POOL_SIZE = 8;

    /**
     * NOTE: This is the number of total available cores. On current versions of
     * Android, with devices that use plug-and-play cores, this will return less
     * than the total number of cores. The total number of cores is not
     * available in current Android implementations.
     */
    private static int NUMBER_OF_CORES = Runtime.getRuntime().availableProcessors();

    // A queue of Runnables for the image download pool
    private final BlockingQueue<Runnable> mObjectDetetionWorkQueue;

    // A queue of Runnables for the image decoding pool
    private final BlockingQueue<Runnable> mFrameTrackingWorkQueue;

    // A queue of PhotoManager tasks. Tasks are handed to a ThreadPool.
    private final BlockingQueue<ProcessTask> mProcessTaskWorkQueue;

    // A managed pool of background download threads
    private final ThreadPoolExecutor mObjectDetetionThreadPool;

    // A managed pool of background decoder threads
    private final ThreadPoolExecutor mFrameTrackingThreadPool;

    // An object that manages Messages in a Thread
    private Handler mHandler;

    // A single instance of PhotoManager, used to implement the singleton pattern
    private static ProcessManager sInstance = null;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private Matrix frameToInputTransform;
    private Matrix inputToFrameTransform;

    private Matrix inputToCropTransform;
    private Matrix cropToInputTransform;

    // A static block that sets class fields
    static {

        // The time unit for "keep alive" is in seconds
        KEEP_ALIVE_TIME_UNIT = TimeUnit.SECONDS;

        // Creates a single static instance of PhotoManager
        sInstance = new ProcessManager();
    }
    /**
     * Constructs the work queues and thread pools used to download and decode images.
     */
    private ProcessManager() {

        /*
         * Creates a work queue for the pool of Thread objects used for object detection, using a linked
         * list queue that blocks when the queue is empty.
         */
        mObjectDetetionWorkQueue = new LinkedBlockingQueue<Runnable>();

        /*
         * Creates a work queue for the pool of Thread objects used for frame tracking, using a linked
         * list queue that blocks when the queue is empty.
         */
        mFrameTrackingWorkQueue = new LinkedBlockingQueue<Runnable>();

        /*
         * Creates a work queue for the set of task objects that control object detection and
         * frame tracking, using a linked list queue that blocks when the queue is empty.
         */
        mProcessTaskWorkQueue = new LinkedBlockingQueue<ProcessTask>();

        /*
         * Creates a new pool of Thread objects for the object detection work queue
         */
        mObjectDetetionThreadPool = new ThreadPoolExecutor(CORE_POOL_SIZE, MAXIMUM_POOL_SIZE,
                KEEP_ALIVE_TIME, KEEP_ALIVE_TIME_UNIT, mObjectDetetionWorkQueue);

        /*
         * Creates a new pool of Thread objects for the frame tracking work queue
         */
        mFrameTrackingThreadPool = new ThreadPoolExecutor(NUMBER_OF_CORES, NUMBER_OF_CORES,
                KEEP_ALIVE_TIME, KEEP_ALIVE_TIME_UNIT, mFrameTrackingWorkQueue);

        /*
         * Instantiates a new anonymous Handler object and defines its
         * handleMessage() method. The Handler *must* run on the UI thread, because it moves photo
         * Bitmaps from the PhotoTask object to the View object.
         * To force the Handler to run on the UI thread, it's defined as part of the PhotoManager
         * constructor. The constructor is invoked when the class is first referenced, and that
         * happens when the View invokes startDownload. Since the View runs on the UI Thread, so
         * does the constructor and the Handler.
         */
        mHandler = new Handler(Looper.getMainLooper()){

            /*
             * handleMessage() defines the operations to perform when the
             * Handler receives a new Message to process.
             */
            @Override
            public void handleMessage(Message inputMessage) {

                // Gets the image task from the incoming Message object.
                ProcessTask processTask = (ProcessTask) inputMessage.obj;

                MultiBoxTracker tracker = processTask.getTracker();

                byte[] luminanceCopy = processTask.getLuminanceCopy();
                List<Classifier.Recognition> mappedRecognitions = processTask.getResults();
                long currentFrame = processTask.getCurrentFrame();

                switch (inputMessage.what) {

                    // If the download has started, sets background color to dark green
                    case DETECTION_STARTED:
                        //localView.setStatusResource(R.drawable.imagedownloading);
                        break;

                    /*
                     * If the download is complete, but the decode is waiting, sets the
                     * background color to golden yellow
                     */
                    case DETECTION_COMPLETE:

                        if (mappedRecognitions != null && tracker!=null) {
                            tracker.trackResults(mappedRecognitions, luminanceCopy, currentFrame);
                        }
                        recycleTask(processTask);

                        break;
                    // If the decode has started, sets background color to orange
                    case TRACKING_STARTED:
                        //localView.setStatusResource(R.drawable.decodedecoding);
                        break;
                    /*
                     * The decoding is done, so this sets the
                     * ImageView's bitmap to the bitmap in the
                     * incoming message
                     */
                    case TRACKING_COMPLETE:
                        //localView.setImageBitmap(photoTask.getImage());
                        recycleTask(processTask);
                        break;
                    // The download failed, sets the background color to dark red
                    case DETECTION_FAILED:
                        //localView.setStatusResource(R.drawable.imagedownloadfailed);
                        // Attempts to re-use the Task object
                        recycleTask(processTask);
                        break;
                    default:
                        // Otherwise, calls the super method
                        super.handleMessage(inputMessage);
                }
            }
        };

    }

    /**
     * Handles state messages for a particular task object
     * @param processTask A task object
     * @param state The state of the task
     */
    @SuppressLint("HandlerLeak")
    public void handleState(ProcessTask processTask, int state) {
        switch (state) {

            // Frame Tracking finished.
            case TRACKING_COMPLETE:

                // Gets a Message object, stores the state in it, and sends it to the Handler
                //Message completeMessage = mHandler.obtainMessage(state, processTask);
                //completeMessage.sendToTarget();
                break;

            // OBJECT DETECTION finished.
            case DETECTION_COMPLETE:

                Message completeMessage = mHandler.obtainMessage(state, processTask);
                completeMessage.sendToTarget();

                // In all other cases, pass along the message without any other action.
            default:
                mHandler.obtainMessage(state, processTask).sendToTarget();
                break;
        }

    }

    /**
     * Returns the PhotoManager object
     * @return The global PhotoManager object
     */
    public static ProcessManager getInstance() {

        return sInstance;
    }

    /**
     * Cancels all Threads in the ThreadPool
     */
    public static void cancelAll() {

        /*
         * Creates an array of tasks that's the same size as the task work queue
         */
        ProcessTask[] taskArray = new ProcessTask[sInstance.mObjectDetetionWorkQueue.size()];

        // Populates the array with the task objects in the queue
        sInstance.mObjectDetetionWorkQueue.toArray(taskArray);

        // Stores the array length in order to iterate over the array
        int taskArraylen = taskArray.length;

        /*
         * Locks on the singleton to ensure that other processes aren't mutating Threads, then
         * iterates over the array of tasks and interrupts the task's current Thread.
         */
        synchronized (sInstance) {

            // Iterates over the array of tasks
            for (int taskArrayIndex = 0; taskArrayIndex < taskArraylen; taskArrayIndex++) {

                // Gets the task's current thread
                Thread thread = taskArray[taskArrayIndex].mThreadThis;

                // if the Thread exists, post an interrupt to it
                if (null != thread) {
                    thread.interrupt();
                }
            }
        }
    }

    /**
     * Stops a frame tracking Thread and removes it from the threadpool
     *
     * @param trackingTask The download task associated with the Thread
     */
    static public void cancelTracking(ProcessTask trackingTask) {

        // If the Thread object still exists and the download matches the specified URL
        if (trackingTask != null) {

            /*
             * Locks on this class to ensure that other processes aren't mutating Threads.
             */
            synchronized (sInstance) {

                // Gets the Thread that the downloader task is running on
                Thread thread = trackingTask.getCurrentThread();

                // If the Thread exists, posts an interrupt to it
                if (null != thread)
                    thread.interrupt();
            }
            /*
             * Removes the download Runnable from the ThreadPool. This opens a Thread in the
             * ThreadPool's work queue, allowing a task in the queue to start.
             */
            sInstance.mFrameTrackingThreadPool.remove(trackingTask.getFrameTrackingRunnable());
        }
    }

    public void startTFDetection(MultiBoxTracker tracker, Classifier detector,
                                        Bitmap inputBitmap, byte[] luminanceCopy, long currentFrame){
        /*
         * Gets a task from the pool of tasks, returning null if the pool is empty
         */
        ProcessTask detectionTask = sInstance.mProcessTaskWorkQueue.poll();

        // If the queue was empty, create a new task instead.
        if (null == detectionTask) {
            detectionTask = new ProcessTask();
        }

        detectionTask.initializeDetector(sInstance, tracker, inputBitmap, luminanceCopy, currentFrame);
        detectionTask.setTFDetector(detector);

        try {
            sInstance.mObjectDetetionThreadPool.execute(detectionTask.getObjectDetectionRunnable());
        } catch (Exception e) {
            e.printStackTrace();
            sInstance.handleState(detectionTask, DETECTION_FAILED);
        }

        sInstance.handleState(detectionTask, DETECTION_STARTED);

        //return detectionTask;

    }

    public void startCVDetection(MultiBoxTracker tracker, CvDetector detector,
                                        Bitmap inputBitmap, AppRandomizer.ReferenceImage refImage,
                                        byte[] luminanceCopy, long currentFrame){
        /*
         * Gets a task from the pool of tasks, returning null if the pool is empty
         */
        ProcessTask detectionTask = sInstance.mProcessTaskWorkQueue.poll();

        // If the queue was empty, create a new task instead.
        if (null == detectionTask) {
            detectionTask = new ProcessTask();
        }

        detectionTask.initializeDetector(sInstance, tracker, inputBitmap, luminanceCopy, currentFrame);
        detectionTask.setCVDetector(detector);

        try {
            //sInstance.handleState(detectionTask, DETECTION_STARTED);
            sInstance.mObjectDetetionThreadPool.execute(detectionTask.getObjectDetectionRunnable());
        } catch (Exception e) {
            e.printStackTrace();
            //sInstance.handleState(detectionTask, DETECTION_FAILED);
        }

        sInstance.handleState(detectionTask, DETECTION_STARTED);

        //return detectionTask;

    }

    /**
     * Recycles tasks by calling their internal recycle() method and then putting them back into
     * the task queue.
     * @param detectionTask The task to recycle
     */
    void recycleTask(ProcessTask detectionTask) {

        // Frees up memory in the task
        detectionTask.recycle();

        // Puts the task object back into the queue for re-use.
        mProcessTaskWorkQueue.offer(detectionTask);
    }

    public boolean isDetectionWorkQueueEmpty(){
        if (this.mObjectDetetionWorkQueue.isEmpty()) return true;
        else return false;
    }

}
