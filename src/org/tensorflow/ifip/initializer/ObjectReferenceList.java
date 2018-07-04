package org.tensorflow.ifip.initializer;

import android.graphics.Bitmap;
import android.graphics.Rect;
import android.graphics.RectF;

import org.opencv.android.Utils;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.xfeatures2d.SIFT;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import static org.tensorflow.ifip.MrCameraActivity.MIN_MATCH_COUNT;

/**
 * Created by deg032 on 20/6/18.
 */

public class ObjectReferenceList {

    private static final ObjectReferenceList instance = new ObjectReferenceList();

    // Private constructor prevents instantiation from other classes
    private ObjectReferenceList() {
    }

    private List<ReferenceObject> list = new ArrayList<>();

    public static ObjectReferenceList getInstance() {
        return instance;
    }

    public List<ReferenceObject> getList() {
        return list;
    }

    public void add(Bitmap inputFrame, String title, RectF location) {

/*        int locX = (int) location.centerX();
        int locY = (int) location.centerY();
        int locW = (int) location.width()/2;
        int locH = (int) location.height()/2;

        if ((locX + locW > inputFrame.getWidth())
                || (locY + locH > inputFrame.getHeight())
                || locX < 0
                || locY < 0) return;*/

        Rect roundedLocation = new Rect();
        location.round(roundedLocation);

        /**
         *  The Math.min operators make sure that x + w <= bitmap.width,
         *  and y + h <= bitmap.height in the createBitmap operation.
         */

        final Bitmap referenceImage = Bitmap.createBitmap(inputFrame,
                Math.abs(roundedLocation.left),
                Math.abs(roundedLocation.top),
                Math.min( (roundedLocation.right - roundedLocation.left),
                        (inputFrame.getWidth()-Math.abs(roundedLocation.left)) ),
                Math.min( (roundedLocation.bottom - roundedLocation.top),
                        (inputFrame.getHeight()-Math.abs(roundedLocation.top)) )
        );

        if (!list.isEmpty()) {
            for (final ReferenceObject listedObject: list){
                /**
                 * If there is a match to an existing object on the list,
                 * stop checking, don't add to list, and, then, return.
                 */
                if (listedObject.getTitle().equals(title))
                    if (objectMatcher(listedObject.getReferenceImage(), referenceImage)) return;
            }
        }

        // A new object will only be added if the list is empty or there is no match.
        list.add(new ReferenceObject(referenceImage, title));
    }

    public void add(ReferenceObject object) {
        list.add(new ReferenceObject(object));
    }

/*    public void refreshList(){
        for (final ReferenceObject object: list){
            // do something
        }
    }*/

    private boolean objectMatcher(Bitmap listedReference, Bitmap freshDetection){

        final SIFT featureDetector = SIFT.create();
        final FlannBasedMatcher descriptorMatcher = FlannBasedMatcher.create();

        Mat refDescriptors = new Mat();
        MatOfKeyPoint refKeypoints = new MatOfKeyPoint();

        Mat refImage = new Mat();
        Utils.bitmapToMat(listedReference, refImage);

        Mat qryDescriptors = new Mat();
        MatOfKeyPoint qryKeypoints = new MatOfKeyPoint();

        Mat qryImage = new Mat();
        Utils.bitmapToMat(freshDetection, qryImage);

        List<MatOfDMatch> matches = new ArrayList<>();

        featureDetector.detect(qryImage, qryKeypoints);
        featureDetector.compute(qryImage, qryKeypoints, qryDescriptors);

        featureDetector.detect(refImage, refKeypoints);
        featureDetector.compute(refImage, refKeypoints, refDescriptors);

        LinkedList<DMatch> good_matches = new LinkedList<>();

        try{
            descriptorMatcher.knnMatch(refDescriptors, qryDescriptors, matches, 2);

            long time = System.currentTimeMillis();

            // ratio test
            for (Iterator<MatOfDMatch> iterator = matches.iterator(); iterator.hasNext();) {
                MatOfDMatch matOfDMatch = iterator.next();
                if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.75) {
                    good_matches.add(matOfDMatch.toArray()[0]);
                }
            }

            long time1 = System.currentTimeMillis();

        } catch (Exception e) {
            e.printStackTrace();
        }

        if (good_matches.size() > MIN_MATCH_COUNT) return true;
        else return false;

    }

    public boolean isSensitive(String targetTitle, Bitmap target) {

        for (final ReferenceObject listedObject: list) {
            if ((listedObject.getTitle().equals(targetTitle)) && (listedObject.getSensitivity()))
                if (objectMatcher(listedObject.getReferenceImage(), target)) return true;
        }

        return false;
    }

    public boolean isSensitive(String targetTitle) {

        for (final ReferenceObject listedObject: list) {
            if ((listedObject.getTitle().equals(targetTitle)) && (listedObject.getSensitivity()))
                return true;
        }

        return false;
    }
}
