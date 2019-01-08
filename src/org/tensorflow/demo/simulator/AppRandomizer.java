package org.tensorflow.demo.simulator;

import android.content.Context;
import android.util.Pair;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.xfeatures2d.SIFT;
import org.tensorflow.demo.R;
import org.tensorflow.demo.env.Logger;

import java.io.Serializable;
import java.sql.Ref;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Created by deg032 on 5/2/18.
 */

public class AppRandomizer implements Randomizer {

    private static final Logger LOGGER = new Logger();

    static final String[] firstMethod = new String[]
            {"TF_DETECTOR","CV_DETECTOR"};

    static final String[] cvMethod = new String[]
            {"SIFT","ORB"};

    static final String[] tfMethod = new String[]
            {"OBJECT_API","CLASSIFIER"};

    static final String[] refnames = new String[]
            {"csiro", "data61", "uhu", "unsw"};

    static final Integer[] drawables = new Integer[]
            {R.drawable.csiro, R.drawable.data61, R.drawable.uhu, R.drawable.unsw};

/*    static final String[] objects = new String[]
            {
                    "laptop", "remote","keyboard", "scissors","cell phone", //office objects
                    "bus", "uhu", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                            "parking meter","bench",//outside objects
                    "bird", "cat", "dog", "horse", "sheep","cow","elephant","bear","zebra",
                            "giraffe", //animal objects
                    "frisbee","skis", "snowboard","sports ball","kite","baseball bat",
                            "baseball glove","skateboard","surfboard",
                            "tennis racket", //sporty objects
                    //"potted plant", "microwave", "oven","toaster","sink","refrigerator","vase",
                    //        "hair drier", "remote", //house objects
                    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                            "cake", "chair","couch", "dining table" //kitchen or food objects
                    //{"person", "bed", "toilet", "laptop", "mouse","keyboard",
                    //        "cell phone"}, //high sensitivity objects
            };*/

    static final String[][] objects = new String[][]
            {
                    {"tv", "laptop", "mouse", "remote","keyboard", "scissors","cell phone",
                            "book"}, //office objects
                    {"bus", "uhu", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                            "parking meter","bench"}, //outside objects
                    {"bird", "cat", "dog", "horse", "sheep","cow","elephant","bear","zebra",
                            "giraffe"}, //animal objects
                    {"frisbee","skis", "snowboard","sports ball","kite","baseball bat",
                            "baseball glove","skateboard","surfboard",
                            "tennis racket"}, //sporty objects
                    {"potted plant", "microwave", "oven","toaster","sink","refrigerator","vase",
                            "hair drier", "tv", "remote", "toilet", "bed"}, //house objects
                    {"bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
                            "sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                            "cake", "chair","couch", "dining table"}, //kitchen or food objects
                    {"person", "bed", "toilet", "laptop", "mouse","keyboard",
                            "cell phone"} //high sensitivity objects
            };

    public class ReferenceImage {

        private String RefName;
        private Mat RefImageMat;
        private MatOfKeyPoint RefKeyPoints;
        private Mat RefDescriptors;

        public String getRefName() {
            return RefName;
        }

        public Mat getRefImageMat() {
            return RefImageMat;
        }

        public MatOfKeyPoint getRefKeyPoints() {
            return RefKeyPoints;
        }

        public Mat getRefDescriptors() {
            return RefDescriptors;
        }

        public void setRefName(String refName) { RefName = refName; }

        public void setRefImageMat(Mat refImageMat) {
            RefImageMat = refImageMat;
        }

        public void setRefDescriptors(Mat refDescriptors) {
            RefDescriptors = refDescriptors;
        }

        public void setRefKeyPoints(MatOfKeyPoint refKeyPoints) {
            RefKeyPoints = refKeyPoints;
        }
    }

    public static Randomizer create(){
        return new AppRandomizer();
    }

    public List<App> appGenerator(Context context, int numberOfApps){
        Random rnd = new Random();
        final List<App> appList = new ArrayList<>(numberOfApps);

        for (int i = 0; i < numberOfApps ; i++){

            // default to TF
            Integer first = rnd.nextInt(firstMethod.length);
            String secondMethod;
            ReferenceImage reference = new ReferenceImage();

            if (first == 0) {
                secondMethod = tfMethod[rnd.nextInt(tfMethod.length)];
            } else {
                secondMethod = cvMethod[rnd.nextInt(cvMethod.length)];
                //secondMethod = "ORB"; // always uses ORB.
                //secondMethod = "SIFT"; // always uses SIFT.
                reference = cvReferenceImage(context, secondMethod);
            }

            Pair<String,String> method = new Pair<>(firstMethod[first],secondMethod);
            String[] objectsOfInterest = objects[rnd.nextInt(objects.length)];
            String name = method.first + "_" + method.second + "_" + Integer.toString(i);
            App app = new App(i,name,method,objectsOfInterest,reference);
            appList.add(app);
        }

        return appList;
    }

    public List<App> fixedAppGenerator(Context context, int numberOfApps){
        Random rnd = new Random();
        final List<App> appList = new ArrayList<>(numberOfApps);

        Integer[][] fixedMethods = new Integer[][]
                {
                        {0,0}, //tf, od
                        {1,0}, //cv, sift
                        {1,1}, //cv, orb
                        {0,1}, //tf, cl
                        {1,0}, //cv, sift
                        {1,1} //cv, orb
                };

        for (int i = 0; i < numberOfApps ; i++){

            Integer first = fixedMethods[i%(fixedMethods.length)][0];
            Integer second = fixedMethods[i%(fixedMethods.length)][1];
            String secondMethod;
            ReferenceImage reference = new ReferenceImage();

            if (first == 0) {
                secondMethod = tfMethod[second];
            } else {
                secondMethod = cvMethod[second];
                //secondMethod = "ORB"; // always uses ORB.
                //secondMethod = "SIFT"; // always uses SIFT.
                reference = cvReferenceImage(context, secondMethod);
            }

            Pair<String,String> method = new Pair<>(firstMethod[first],secondMethod);
            String[] objectsOfInterest = objects[rnd.nextInt(objects.length)];
            String name = method.first + "_" + method.second + "_" + Integer.toString(i);
            App app = new App(i,name,method,objectsOfInterest,reference);
            appList.add(app);
        }

        return appList;
    }

    private ReferenceImage cvReferenceImage(Context context, String method) {
        /** Extract the reference SIFT features */
        final ReferenceImage reference = new ReferenceImage();

        MatOfKeyPoint refKeyPoints = new MatOfKeyPoint();
        Mat refDescriptors = new Mat();
        Mat refImageMat = new Mat();

        Integer drawable = 2; //default to UHU ref, new Random().nextInt(drawables.length);

        long time = System.currentTimeMillis();

        if (method == "SIFT") { // If the internal CV Method uses SIFT
            try {
                refImageMat = Utils.loadResource(context, drawables[drawable],
                        Imgcodecs.CV_LOAD_IMAGE_COLOR);

                SIFT mFeatureDetector = SIFT.create();

                LOGGER.i("Using "+ method +", Height: " + Integer.toString(refImageMat.height())
                        + ", Width: " + Integer.toString(refImageMat.width()));

                mFeatureDetector.detect(refImageMat, refKeyPoints);
                mFeatureDetector.compute(refImageMat,refKeyPoints, refDescriptors);
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        } else { // If the internal CV Method uses ORB
            try {
                refImageMat = Utils.loadResource(context, drawables[drawable],
                        Imgcodecs.CV_LOAD_IMAGE_COLOR);

                ORB mFeatureDetector = ORB.create();

                LOGGER.i("Using " + method + ", Height: " + Integer.toString(refImageMat.height())
                        + ", Width: " + Integer.toString(refImageMat.width()));

                mFeatureDetector.detect(refImageMat, refKeyPoints);
                mFeatureDetector.compute(refImageMat, refKeyPoints, refDescriptors);

            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        LOGGER.i("Time to process " + (System.currentTimeMillis() - time) +
                ", Number of key points: " + refKeyPoints.toArray().length);

        reference.setRefName(refnames[drawable]);
        reference.setRefImageMat(refImageMat);
        reference.setRefDescriptors(refDescriptors);
        reference.setRefKeyPoints(refKeyPoints);

        return reference;
    }

}
