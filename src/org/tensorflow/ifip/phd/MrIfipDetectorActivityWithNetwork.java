package org.tensorflow.ifip.phd;

/**
 * Created by deg032 on 5/2/18.
 *
 * This is the first attempt to create a 'local' abstraction overlay that presents an object-level
 * fine-grained access control.
 *
 * [6-Feb-2018] Now, utilizing an objectManager that handles live detected objects and which
 * currently running apps have access to them.
 */

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import org.tensorflow.ifip.Classifier;
import org.tensorflow.ifip.MrCameraActivity;
import org.tensorflow.ifip.OverlayView;
import org.tensorflow.ifip.OverlayView.DrawCallback;
import org.tensorflow.ifip.R;
import org.tensorflow.ifip.TensorFlowMultiBoxDetector;
import org.tensorflow.ifip.TensorFlowObjectDetectionAPIModel;
import org.tensorflow.ifip.TensorFlowYoloDetector;
import org.tensorflow.ifip.augmenting.Augmenter;
import org.tensorflow.ifip.env.BorderedText;
import org.tensorflow.ifip.env.ImageUtils;
import org.tensorflow.ifip.env.Logger;
import org.tensorflow.ifip.network.RemoteDetector;
import org.tensorflow.ifip.phd.detector.cv.CvDetector;
import org.tensorflow.ifip.phd.detector.cv.OrbDetector;
import org.tensorflow.ifip.phd.detector.cv.SiftDetector;
import org.tensorflow.ifip.simulator.App;
import org.tensorflow.ifip.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

/**
 * An activity that follows Tensorflow's demo DetectorActivity class as template and implements
 * classical visual detection using OpenCV.
 */
public class MrIfipDetectorActivityWithNetwork extends MrCameraActivity {

    private static final Logger LOGGER = new Logger();

    private int captureCount = 0;
    private int secrecyHit = 0;

    // Configuration values for the prepackaged multibox model.
    private static final int MB_INPUT_SIZE = 224;
    private static final int MB_IMAGE_MEAN = 128;
    private static final float MB_IMAGE_STD = 128;
    private static final String MB_INPUT_NAME = "ResizeBilinear";
    private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
    private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
    private static final String MB_MODEL_FILE = "file:///android_asset/multibox_model.pb";
    private static final String MB_LOCATION_FILE =
            "file:///android_asset/multibox_location_priors.txt";

    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final String TF_OD_API_MODEL_FILE =
            "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";

    // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
    // must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
    // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
    // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
    private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.
    private enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

    // Configuration values for the regular non-boxing classifier.
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Classifier detector; //for TF detection
    private Classifier classifier; //for TF classification
    private Classifier remoteDetector;
    private SiftDetector siftDetector; //for OpenCV detection
    private OrbDetector orbDetector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;
    private Bitmap inputBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private Matrix frameToInputTransform;
    private Matrix inputToFrameTransform;

    private Matrix inputToCropTransform;
    private Matrix cropToInputTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    OverlayView trackingOverlay;

    private OverlayView augmentedOverlay;
    private Augmenter augmenter;

    static {
        LOGGER.i("DataGathering, OperatingMode, Image number, No of apps, Size, Total Time, Detection Time");
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        inputSizeWidth = size.getWidth();
        inputSizeHeight = size.getHeight();
        inputSize = Math.min(inputSizeWidth, inputSizeHeight);

        LOGGER.i("onPreviewSizeChosen: operatingMode = " + operatingMode);

        tracker = new MultiBoxTracker(this);
        augmenter = new Augmenter();

        // setting up a TF detector (a TF OD type)
        int cropSize = TF_OD_API_INPUT_SIZE;
        if (MODE == MrIfipDetectorActivityWithNetwork.DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            getAssets(),
                            YOLO_MODEL_FILE,
                            inputSize,//YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            inputSize = YOLO_INPUT_SIZE;
        } else if (MODE == MrIfipDetectorActivityWithNetwork.DetectorMode.MULTIBOX) {
            detector =
                    TensorFlowMultiBoxDetector.create(
                            getAssets(),
                            MB_MODEL_FILE,
                            MB_LOCATION_FILE,
                            MB_IMAGE_MEAN,
                            MB_IMAGE_STD,
                            MB_INPUT_NAME,
                            MB_OUTPUT_LOCATIONS_NAME,
                            MB_OUTPUT_SCORES_NAME);
            inputSize = MB_INPUT_SIZE;
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        getAssets(),
                        TF_OD_API_MODEL_FILE,
                        TF_OD_API_LABELS_FILE,
                        inputSize//TF_OD_API_INPUT_SIZE
                );
            } catch (final IOException e) {
                LOGGER.e("Exception initializing classifier!", e);
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }
        }

        /*// setting up a TF classifier
        classifier =
                TensorFlowImageClassifier.create(
                        getAssets(),
                        MODEL_FILE,
                        LABEL_FILE,
                        INPUT_SIZE,
                        IMAGE_MEAN,
                        IMAGE_STD,
                        INPUT_NAME,
                        OUTPUT_NAME);*/

        /**
         * Inserted the line below for the OpenCV Detector.
         */

        siftDetector = new SiftDetector();
        orbDetector = new OrbDetector();

        if (NetworkMode.equals("REMOTE_PROCESS"))
            remoteDetector = RemoteDetector.create(remoteUrl, operatingMode);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        inputToCropTransform =
                ImageUtils.getTransformationMatrix(
                        inputSizeWidth, inputSizeHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToInputTransform = new Matrix();
        inputToCropTransform.invert(cropToInputTransform);

        frameToInputTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        inputSizeWidth, inputSizeHeight,
                        sensorOrientation, MAINTAIN_ASPECT);

        inputToFrameTransform = new Matrix();
        frameToInputTransform.invert(inputToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        /**
         * This following addCallback is declared in the CameraActivity class and is another method
         * implementation of the addCallback from the OverlayView. Also, this is the one that
         * draws on the debug_overlay, whereas the previous one is for the tracking_overlay.
         */
        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        //if (!isDebug()) {
                        //    return;
                        //}
                        final Bitmap copy = cropCopyBitmap;
                        if (copy == null) {
                            return;
                        }

                        final int backgroundColor = Color.argb(100, 0, 0, 0);
                        canvas.drawColor(backgroundColor);

                        final Matrix matrix = new Matrix();
                        final float scaleFactor = 2;
                        matrix.postScale(scaleFactor, scaleFactor);
                        matrix.postTranslate(
                                canvas.getWidth() - copy.getWidth() * scaleFactor,
                                canvas.getHeight() - copy.getHeight() * scaleFactor);
                        canvas.drawBitmap(copy, matrix, new Paint());

                        final Vector<String> lines = new Vector<String>();
                        if (detector != null) {
                            final String statString = detector.getStatString();
                            final String[] statLines = statString.split("\n");
                            for (final String line : statLines) {
                                lines.add(line);
                            }
                        }
                        lines.add("");

                        lines.add("Running " + singletonAppList.getList().size() + " apps");
                        lines.add("Preview Frame: " + previewWidth + "x" + previewHeight);
                        lines.add("Processed Frame: " + inputSize + "x" + inputSize);
                        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                        //lines.add("Rotation: " + sensorOrientation);
                        lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                    }
                });

        /**
         * We have created this augmented overlay in which augmentations are to be rendered.
         * Due to the preset canvas type it accesses, for the meantime we use this canvas types to
         * draw. However, future implementations may want to use a GL type View so as to draw GL
         * objects (which has more 3D support than canvas).
         */
        augmentedOverlay = (OverlayView) findViewById(R.id.augmented_overlay);
        augmentedOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        augmenter.drawAugmentations(canvas);
                    }
                });

    }

    private void sharedAbstraction(){
        //if (isNetworkConnected())
            manager.
                refreshListFromNetwork(mNetworkFragment,receiveFlag);
        //else manager.refreshList();
    }


    private boolean isNetworkConnected() {
        ConnectivityManager connMgr = (ConnectivityManager)
                getSystemService(Context.CONNECTIVITY_SERVICE); // 1
        NetworkInfo networkInfo = connMgr.getActiveNetworkInfo(); // 2
        return networkInfo != null && networkInfo.isConnected(); // 3
    }

    private void noConnection(){
        new AlertDialog.Builder(this)
                .setTitle("No Internet Connection")
                .setMessage("It looks like your internet connection is off. Please turn it " +
                        "on and try again")
                .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                    }
                }).setIcon(android.R.drawable.ic_dialog_alert).show();
    }


    @Override // part of MrCameraActivity
    protected void processImage() {

/*        if (captureCount >= CAPTURE_TIMEOUT) {
            if (instanceCount >= INSTANCE_TIMEOUT) {
                if (appCount < 10) {
                    appCount++;
                } else return;
                instanceCount = 0;
            } else instanceCount++;

            appList = ultimateAppList.get(appCount).get(instanceCount).first;
            appListText = ultimateAppList.get(appCount).get(instanceCount).second;
            captureCount = 0;
        }*/

        if (fastDebug) if (captureCount > CAPTURE_TIMEOUT) {
            long sum = 0;
            for (long d : overallTimes) sum += d;
            double averageOverall = 1.0d * sum / overallTimes.length;

            sum = 0;
            for (long d : detectionTimes) sum += d;
            double averageDetection = 1.0d * sum / detectionTimes.length;

            LOGGER.i("DataGatheringAverage, %d, %d, %d, %f, %f",
                    captureCount, appList.size(),inputSize, averageOverall, averageDetection);

/*            if (logWriter!=null) {
                try {
                    logWriter.write("DataGathering," + captureCount +","
                            + appList.size() + "," + inputSize +"," + averageOverall + ","
                            + averageDetection);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }*/


            return;
        }


        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();

        // Usually, this onFrame method below doesn't really happen as you would see in the toast
        // message that appears when you start up this detector app.
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }

        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        inputBitmap = Bitmap.createScaledBitmap(rgbFrameBitmap,inputSize, inputSize,true);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }

        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                            case MULTIBOX:
                                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                                break;
                            case YOLO:
                                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                                break;
                        }

                        // final list of recognitions before rendering to the trackingOverlayView
                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<>();

                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        long detectionTime = 0;//SystemClock.uptimeMillis() - startTime;
                        long begin = 0;

                        // This for-loop below performs *detection* AND *transformation* for each
                        // concurrent app that's running.
                        for (final App app : appList) {

                            LOGGER.i("Doing app: " + app.toString());
                            long detect1 = 0;

                            // Implementation of Local Abstraction in Detection: TF and CV
                            List<Classifier.Recognition> dResults = new ArrayList<>();

                            CvDetector.QueryImage sResult = new CvDetector.QueryImage();
                            CvDetector.QueryImage oResult = new CvDetector.QueryImage();

                            //getting objects of Interest of an app
                            List<String> objectsOfInterest = Arrays.asList(app.getObjectsOfInterest());

                            final List<Classifier.Recognition> appResults =
                                    new LinkedList<>(); // collection of results per app

                            Integer localHit = 0;
                            Integer localSecrecyHit = 0;

                            if (NetworkMode.equals("REMOTE_PROCESS")) {
                                LOGGER.d("Detection done remotely.");

                                begin = SystemClock.uptimeMillis();
                                dResults = remoteDetector.recognizeImage(inputBitmap);
                                detect1 = SystemClock.uptimeMillis()-begin;

                                if (dResults != null) {
                                    //transformation
                                    for (final Classifier.Recognition dResult : dResults) {
                                        final RectF location = dResult.getLocation();
                                        if (location != null && dResult.getConfidence() >= minimumConfidence) {
                                            inputToCropTransform.mapRect(location);
                                            canvas.drawRect(location, paint);

                                            if (Arrays.asList(secretObjects).contains(dResult.getTitle())){
                                                localSecrecyHit = 1;
                                                //continue; //Don't overlay if object is secret.
                                            } else if (objectsOfInterest.contains(dResult.getTitle())){ // case 3
                                                localHit = 1;
                                            }

                                            cropToFrameTransform.mapRect(location);
                                            dResult.setLocation(location);
                                            appResults.add(dResult);

                                            //localHit = 1;

                                        }
                                    }

                                } // if null, no result. No transformation.

                            } else {
                                LOGGER.d("Detection done locally.");

                                /**
                                 * Abstraction-style processing: will perform all necessary processing
                                 * as specified by the task list (i.e. app list).
                                 */
                                /*if (appListText.contains("TF"))
                                    dResults = detector.recognizeImage(inputBitmap);
                                if (appListText.contains("SIFT"))
                                    sResult = siftDetector.imageDetector(inputBitmap);
                                if (appListText.contains("ORB"))
                                    oResult = orbDetector.imageDetector(inputBitmap);*/

                                List<Classifier.Recognition> results = new ArrayList<>();

                                CvDetector.Recognition result = new CvDetector.Recognition();
                                Path locationPath;
                                RectF locationRectF;

                                Classifier.Recognition cvDetection;

                                switch (operatingMode) {

                                    case "TF":

                                        begin = SystemClock.uptimeMillis();
                                        results = detector.recognizeImage(inputBitmap); // no classifier
                                        detect1 = SystemClock.uptimeMillis()-begin;

                                        //transformation
                                        for (final Classifier.Recognition dResult : results) {
                                            final RectF location = dResult.getLocation();
                                            if (location != null && dResult.getConfidence() >= minimumConfidence) {
                                                inputToCropTransform.mapRect(location);
                                                canvas.drawRect(location, paint);

                                                if (Arrays.asList(secretObjects).contains(dResult.getTitle())){
                                                    localSecrecyHit = 1;
                                                    //continue; //Don't overlay if object is secret.
                                                } else if (objectsOfInterest.contains(dResult.getTitle())){ // case 3
                                                    localHit = 1;
                                                }

                                                cropToFrameTransform.mapRect(location);
                                                dResult.setLocation(location);
                                                appResults.add(dResult);

                                                //localHit = 1;

                                            }
                                        }

                                        break;

                                    case "SIFT":

                                        begin = SystemClock.uptimeMillis();
                                        result = siftDetector.imageDetector(inputBitmap, app.getReference());
                                        detect1 = SystemClock.uptimeMillis()-begin;

                                        if (result == null) break;

                                        localHit = 1;

                                        result.setTitle(app.getName());

                                        locationPath = result.getLocation().first;
                                        locationPath.transform(inputToCropTransform);
                                        canvas.drawPath(locationPath, paint);

                                        locationRectF = result.getLocation().second;
                                        inputToCropTransform.mapRect(locationRectF);
                                        cropToFrameTransform.mapRect(locationRectF);

                                        cvDetection = new Classifier.
                                                Recognition(app.getMethod().second, result.getTitle(),
                                                minimumConfidence, result.getLocation().second);
                                        appResults.add(cvDetection);

                                        break;

                                    case "ORB":

                                        begin = SystemClock.uptimeMillis();
                                        result = orbDetector.imageDetector(inputBitmap, app.getReference());
                                        detect1 = SystemClock.uptimeMillis()-begin;

                                        if (result == null) break;

                                        localHit = 1;

                                        result.setTitle(app.getName());

                                        locationPath = result.getLocation().first;
                                        locationPath.transform(inputToCropTransform);
                                        canvas.drawPath(locationPath, paint);

                                        locationRectF = result.getLocation().second;
                                        inputToCropTransform.mapRect(locationRectF);
                                        cropToFrameTransform.mapRect(locationRectF);

                                        cvDetection = new Classifier.
                                                Recognition(app.getMethod().second, result.getTitle(),
                                                minimumConfidence, result.getLocation().second);
                                        appResults.add(cvDetection);

                                        break;

                                }

                            }

                            // Measuring local detection time per app
                            long detect2 = detectionTime;
                            detectionTime = detect2 + detect1;

                            // Measuring local hits for utility calculation
                            final Integer previousHit = utilityHit;
                            utilityHit = previousHit + localHit;

                            final Integer previousSecretHit = secrecyHit;
                            secrecyHit = previousSecretHit + localSecrecyHit;

                            /*app.addCallback(
                                    new App.AppCallback() {
                                        @Override
                                        public void appCallback() {
                                            for (final Classifier.Recognition result: appResults) {
                                                app.process(result, currTimestamp);
                                            }
                                        }
                                    }
                            );*/

                            mappedRecognitions.addAll(appResults);

                        }

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        // pretty much rendering
                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        //augmenter.trackResults(luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();

                        requestRender();
                        computingDetection = false;

                        final long overallTime = SystemClock.uptimeMillis() - startTime;

                        LOGGER.i("DataGathering, %s_%s, %d, %d, %d, %d, %d", //%d, %d",
                                operatingMode, NetworkMode, captureCount, appList.size(),inputSize, overallTime, detectionTime);

                        if (fastDebug) {
                            overallTimes[captureCount] = overallTime;
                            detectionTimes[captureCount] = detectionTime;
                        }

/*                        if (logWriter!=null) {
                            try {
                                logWriter.write("DataGathering," + captureCount +","
                                        + appList.size() + "," + inputSize +"," + overallTime + ","
                                        + detectionTime);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }*/


                        ++captureCount;
                    }
                });

        //sharedAbstraction();

    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_augmented;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return new Size(inputSize, inputSize);
        //default to --> return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onSetDebug(final boolean debug) {
        detector.enableStatLogging(debug);
    }
}
