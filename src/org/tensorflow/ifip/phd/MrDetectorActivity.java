package org.tensorflow.ifip.phd;

/**
 * Created by deg032 on 25/1/18.
 *
 * This activity emulates multiple concurrent applications/services accessing the captured frame,
 * and individually (and separately) performing their corresponding detections (see processImage())
 * directly on the captured frame.
 */

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
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import org.tensorflow.ifip.MrCameraActivity;
import org.tensorflow.ifip.Classifier;
import org.tensorflow.ifip.OverlayView;
import org.tensorflow.ifip.OverlayView.DrawCallback;
import org.tensorflow.ifip.R;
import org.tensorflow.ifip.TensorFlowMultiBoxDetector;
import org.tensorflow.ifip.TensorFlowObjectDetectionAPIModel;
import org.tensorflow.ifip.TensorFlowYoloDetector;
import org.tensorflow.ifip.phd.detector.cv.CvDetector;
import org.tensorflow.ifip.phd.detector.cv.OrbDetector;
import org.tensorflow.ifip.phd.detector.cv.SiftDetector;
import org.tensorflow.ifip.env.BorderedText;
import org.tensorflow.ifip.env.ImageUtils;
import org.tensorflow.ifip.env.Logger;
import org.tensorflow.ifip.simulator.App;
import org.tensorflow.ifip.tracking.MultiBoxTracker;

import org.tensorflow.ifip.augmenting.Augmenter;
import org.tensorflow.ifip.tracking.MultiBoxTrackerDemo;

/**
 * An activity that follows Tensorflow's demo DetectorActivity class as template and implements
 * classical visual detection using OpenCV.
 */
public class MrDetectorActivity extends MrCameraActivity implements OnImageAvailableListener {
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

    // Configuration values for the regular non-boxing classifier.
    private static final int INPUT_SIZE = 224;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";

    private static final String MODEL_FILE = "file:///android_asset/tensorflow_inception_graph.pb";
    private static final String LABEL_FILE =
            "file:///android_asset/imagenet_comp_graph_label_strings.txt";


    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;

    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Classifier detector; //for TF detection
    private Classifier classifier; //for TF classification
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

    private MultiBoxTrackerDemo tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    OverlayView trackingOverlay;

    private OverlayView augmentedOverlay;
    private Augmenter augmenter;

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTrackerDemo(this);

        //For non-CV tracking, comment out the line below.
        augmenter = new Augmenter();

        // setting up a TF detectMarkers (a TF OD type)
        int cropSize = TF_OD_API_INPUT_SIZE;
        if (MODE == DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            getAssets(),
                            YOLO_MODEL_FILE,
                            inputSize, //YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            cropSize = YOLO_INPUT_SIZE;
        } else if (MODE == DetectorMode.MULTIBOX) {
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
            cropSize = MB_INPUT_SIZE;
            LOGGER.i("Created detectMarkers using Multi-box Detector");
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        getAssets(),
                        TF_OD_API_MODEL_FILE,
                        TF_OD_API_LABELS_FILE,
                        inputSize//TF_OD_API_INPUT_SIZE
                        );
                cropSize = TF_OD_API_INPUT_SIZE;
                LOGGER.i("Created detectMarkers using Object Detector");
            } catch (final IOException e) {
                LOGGER.e("Exception initializing classifier!", e);
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }
        }

        // setting up a TF classifier
       /* classifier =
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

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
        //inputBitmap = Bitmap.createBitmap(inputSize, inputSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        inputToCropTransform =
                ImageUtils.getTransformationMatrix(
                        inputSize, inputSize,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToInputTransform = new Matrix();
        inputToCropTransform.invert(cropToInputTransform);

        frameToInputTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        inputSize, inputSize,
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
         * This following addCallback is declared in the MrThreadedCameraActivity class and is another method
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
                        lines.add("Frame: " + previewWidth + "x" + previewHeight);
                        lines.add("Processed Frame: " + inputBitmap.getWidth() + "x" + inputBitmap.getWidth());
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

    @Override
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

            LOGGER.i("DataGatheringAverage, %d, %d, %f, %f, %d",
                    appList.size(),inputSize, averageOverall, averageDetection,
                    utilityHit);
            if (logWriter!=null) {

                try{
                    logWriter.write("DataGathering," + captureCount +","
                            + appList.size() + "," + inputSize +"," + averageOverall + ","
                            + averageDetection);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            return;
        }

        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        //final byte[] mBytes = getLuminance();//getImageMat();
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
        inputBitmap = Bitmap.createScaledBitmap(rgbFrameBitmap,inputSize,inputSize,true);

        // Usually, this onFrame method below doesn't really happen as you would see in the toast
        // message that appears when you start up this detectMarkers app.
        tracker.FrameTracker(
                previewWidth,
                previewHeight,
                sensorOrientation,
                rgbFrameBitmap, //inputBitmap,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }

        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

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

        //final Canvas inputCanvas = new Canvas(inputBitmap);
        //inputCanvas.drawBitmap(rgbFrameBitmap, frameToInputTransform, null);

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {

                        tracker.setFrame(rgbFrameBitmap);

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

                        // final list of recognitions before rendering ot the trackingOverlayView
                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<>();

                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        long detectionTime = 0;
                        long begin = 0;

                        // This for-loop below performs detection and transformation for each
                        // concurrent app that's running. Furthermore, the detection process can be
                        // performed remotely (instead of locally) by an app's associated remote
                        // supporting servers.
                        for (final App app : appList) {

                            LOGGER.i("Doing app: " + app.toString());

                            long detect1 = 0;

                            final List<Classifier.Recognition> appResults =
                                    new LinkedList<>(); // collection of results per app
                            List<String> objectsOfInterest = Arrays.asList(app.getObjectsOfInterest());

                            Integer localHit = 0;
                            Integer localSecrecyHit = 0;

                            switch (app.getMethod().first) {
                                case "TF_DETECTOR":
                                    List<Classifier.Recognition> results = new ArrayList<>();

                                    /*switch (app.getMethod().second){
                                        case "MULTIBOX":
                                            //detection
                                            results = detectMarkers.recognizeImage(croppedBitmap);

                                            break;

                                        case "CLASSIFIER":
                                            // detection
                                            results = classifier.recognizeImage(croppedBitmap);

                                            break;
                                    }*/
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

                                case "CV_DETECTOR":

                                    begin = SystemClock.uptimeMillis();
                                    CvDetector.Recognition result = new CvDetector.Recognition();

                                    switch (app.getMethod().second) {
                                        case "SIFT":
                                            //detection
                                            result = siftDetector.imageDetector(inputBitmap, app.getReference());
                                            break;
                                        case "ORB":
                                            //detection
                                            result = orbDetector.imageDetector(inputBitmap, app.getReference());
                                            break;
                                    }
                                    detect1 = SystemClock.uptimeMillis()-begin;

                                    if (result == null) break;

                                    localHit = 1;

                                    result.setTitle(app.getName());

                                    Path locationPath = result.getLocation().first;
                                    locationPath.transform(inputToCropTransform);
                                    canvas.drawPath(locationPath, paint);

                                    RectF locationRectF = result.getLocation().second;
                                    inputToCropTransform.mapRect(locationRectF);
                                    cropToFrameTransform.mapRect(locationRectF);
                                    Classifier.Recognition cvDetection = new Classifier.
                                            Recognition(app.getMethod().second, result.getTitle(),
                                            minimumConfidence, result.getLocation().second);
                                    appResults.add(cvDetection);

                                    break;

                            }

                            // Measuring local detection time per app
                            long detect2 = detectionTime;
                            detectionTime = detect2 + detect1;

                            // Measuring local hits for utility calculation
                            final Integer previousHit = utilityHit;
                            utilityHit = previousHit + localHit;

                            final Integer previousSecretHit = secrecyHit;
                            secrecyHit = previousSecretHit + localSecrecyHit;

                            app.addCallback(
                                    new App.AppCallback() {
                                        @Override
                                        public void appCallback() {
                                            for (final Classifier.Recognition result: appResults) {
                                                app.process(result, currTimestamp);
                                            }
                                        }
                                    }
                            );

                            mappedRecognitions.addAll(appResults);

                        }

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        // pretty much rendering
                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        augmenter.trackResults(luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();

                        requestRender();
                        computingDetection = false;
                        tracker.setInitialize(false);

                        final long overallTime = SystemClock.uptimeMillis() - startTime;

                        LOGGER.i("DataGathering, %d, %d, %d, %d, %d, %d, %d",
                                captureCount, appList.size(),inputSize,overallTime, detectionTime,
                                utilityHit, secrecyHit);
                        if (fastDebug) {
                            overallTimes[captureCount] = overallTime;
                            detectionTimes[captureCount] = detectionTime;
                        }

                        if (logWriter!=null) {

                            try {
                                logWriter.write("DataGathering," + captureCount +","
                                        + appList.size() + "," + inputSize +"," + overallTime + ","
                                        + detectionTime);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }

                        ++captureCount;
                    }
                });

    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_augmented;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onSetDebug(final boolean debug) {
        detector.enableStatLogging(debug);
    }
}
