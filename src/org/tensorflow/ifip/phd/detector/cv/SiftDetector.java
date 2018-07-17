package org.tensorflow.ifip.phd.detector.cv;

/**
 * Created by deg032 on 1/2/18.
 */

import android.graphics.Bitmap;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.Pair;

import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.FlannBasedMatcher;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SIFT;
import org.tensorflow.ifip.env.Logger;
import org.tensorflow.ifip.simulator.AppRandomizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import static org.tensorflow.ifip.MrCameraActivity.MIN_MATCH_COUNT;

public class SiftDetector implements CvDetector{

    private static final Logger LOGGER = new Logger();

    // This is a separate imageDetector which only extracts the OpenCV query image matrix and
    // associated descriptor and keypoint matrix using SIFT.
    public static CvDetector create(){
        final SiftDetector d = new SiftDetector();
        return d;
    }

    @Override
    public QueryImage imageDetector(Bitmap bitmap){

        final SIFT mFeatureDetector = SIFT.create();
        final MatOfKeyPoint mKeyPoints = new MatOfKeyPoint();
        final Mat mDescriptors = new Mat();

        long startTime = System.currentTimeMillis();

        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap,mat);

        LOGGER.d("Matrix has width: " + Integer.toString(mat.width())
                + " and height: " + Integer.toString(mat.height()));

        try {
            mFeatureDetector.detect(mat, mKeyPoints);
            mFeatureDetector.compute(mat, mKeyPoints, mDescriptors);
            LOGGER.d("Time to Extract locally: " + Long.toString((System.currentTimeMillis() - startTime))
                    + ", Number of Key points: " + mKeyPoints.toArray().length);

        } catch (Exception e) {
            e.printStackTrace();
            LOGGER.d("Cannot process.");
            return null;
        } /**finally
         } */

        return new QueryImage(mat, mKeyPoints, mDescriptors);
    }

    @Override
    public Recognition imageDetector(Bitmap bitmap, AppRandomizer.ReferenceImage reference) {

        final QueryImage query = imageDetector(bitmap);
        ArrayList<org.opencv.core.Point> scenePoints = new ArrayList<>();

        MatOfKeyPoint keypoints = query.getQryKeyPoints();

        if (0 != keypoints.toArray().length) {
            scenePoints = imageMatcher(query, reference);
            //Imgproc.drawContours(mat, scenePoints, 0, new Scalar(255, 0, 0), 3);
            // for using the draw contours again, please change scenePoints from
            // ArrayList<Point> to List<MatOfPoint>, and change the ImageMatcher
            // return value as well to List<MatOfPoint> type.
        } else {
            LOGGER.d("Cannot process: No key points");
            return null;
        }

        final Path path = new Path();
        final RectF location = new RectF();

        if (scenePoints != null) {
            /**
             * Using path to draw a transformed bounding box.
             */
            path.moveTo((float) scenePoints.get(0).x, (float) scenePoints.get(0).y);
            path.lineTo((float) scenePoints.get(1).x, (float) scenePoints.get(1).y);
            path.lineTo((float) scenePoints.get(2).x, (float) scenePoints.get(2).y);
            path.lineTo((float) scenePoints.get(3).x, (float) scenePoints.get(3).y);
            path.close();

            /**
             * Using RectF to draw a fixed rectangle bounding box.
             */
            float[] xValues = {(float) scenePoints.get(0).x,
                    (float) scenePoints.get(1).x,
                    (float) scenePoints.get(2).x,
                    (float) scenePoints.get(3).x};
            float[] yValues = {(float) scenePoints.get(0).y,
                    (float) scenePoints.get(1).y,
                    (float) scenePoints.get(2).y,
                    (float) scenePoints.get(3).y};
            Arrays.sort(xValues);
            Arrays.sort(yValues);
            location.set(xValues[0], yValues[0], xValues[3], yValues[3]);
        } else return null;

        return new Recognition("",Pair.create(path, location));
    }

    private ArrayList<Point> imageMatcher(QueryImage queryImage, AppRandomizer.ReferenceImage reference){

        ArrayList<org.opencv.core.Point> points = new ArrayList<>();
        List<MatOfPoint> mScenePoints = new ArrayList<>();
        List<MatOfDMatch> matches = new ArrayList<>();
        FlannBasedMatcher descriptorMatcher = FlannBasedMatcher.create();

        Mat refImage = reference.getRefImageMat();
        Mat qryImage = queryImage.getQryImageMat();

        Mat refDescriptors = reference.getRefDescriptors();
        Mat qryDescriptors = queryImage.getQryDescriptors();

        MatOfKeyPoint refKeypoints = reference.getRefKeyPoints();
        MatOfKeyPoint qryKeypoints = queryImage.getQryKeyPoints();

        try{
            descriptorMatcher.knnMatch(refDescriptors, qryDescriptors, matches, 2);

            long time = System.currentTimeMillis();

            // ratio test
            LinkedList<DMatch> good_matches = new LinkedList<>();
            for (Iterator<MatOfDMatch> iterator = matches.iterator(); iterator.hasNext();) {
                MatOfDMatch matOfDMatch = iterator.next();
                if (matOfDMatch.toArray()[0].distance / matOfDMatch.toArray()[1].distance < 0.75) {
                    good_matches.add(matOfDMatch.toArray()[0]);
                }
            }

            long time1 = System.currentTimeMillis();

            if (good_matches.size() > MIN_MATCH_COUNT){

                /** get keypoint coordinates of good matches to find homography and remove outliers
                 * using ransac */
                /** Also, always remember that this is already a transformation process. */

                List<org.opencv.core.Point> refPoints = new ArrayList<>();
                List<org.opencv.core.Point> mPoints = new ArrayList<>();
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
                try {
                    Core.perspectiveTransform(obj_corners,scene_corners,Homog);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                MatOfPoint sceneCorners = new MatOfPoint();
                for (int i=0; i < scene_corners.rows(); i++) {
                    org.opencv.core.Point point = new org.opencv.core.Point();
                    point.set(scene_corners.get(i,0));
                    points.add(point);
                }
                sceneCorners.fromList(points);
                mScenePoints.add(sceneCorners);

                if (Imgproc.contourArea(mScenePoints.get(0)) > (MIN_MATCH_COUNT*MIN_MATCH_COUNT)) {
                    LOGGER.i("Time to Match: " + Long.toString((time1 - time))
                            + ", Number of matches: " + good_matches.size()
                            + " (" + Integer.toString(MIN_MATCH_COUNT) + ")"
                            + ", Time to transform: " + Long.toString((System.currentTimeMillis() - time1)));
                } else {
                    // Transformation is too small or skewed, object probably not in view, or matching
                    // error.
                    LOGGER.i( "Time to Match: " + Long.toString((time1 - time))
                            + ", Object probably not in view even with " + good_matches.size()
                            + " (" + Integer.toString(MIN_MATCH_COUNT) + ") matches.");

                    return null;
                }
                //result = "Enough matches.";
            } else {
                LOGGER.i( "Time to Match: " + Long.toString((System.currentTimeMillis() - time))
                        + ", Not Enough Matches (" + good_matches.size()
                        + "/" + Integer.toString(MIN_MATCH_COUNT) + ")");
                //result = "Not enough matches.";
                return null;
            }

        } catch (Exception e) {
            e.printStackTrace();
            LOGGER.d("Cannot process.");
            return null;
        }

        return points; //mScenePoints; for using drawContours
    }

    // This is a method that just does Matching and returns a Path/RectF if there is a match.
    @Override
    public Recognition getTransformation(QueryImage queryImage,
                                         AppRandomizer.ReferenceImage reference) {

        ArrayList<org.opencv.core.Point> scenePoints = imageMatcher(queryImage, reference);

        final Path path = new Path();
        final RectF location = new RectF();

        if (scenePoints != null) {
            /**
             * Using path to draw a transformed bounding box.
             */
            path.moveTo((float) scenePoints.get(0).x, (float) scenePoints.get(0).y);
            path.lineTo((float) scenePoints.get(1).x, (float) scenePoints.get(1).y);
            path.lineTo((float) scenePoints.get(2).x, (float) scenePoints.get(2).y);
            path.lineTo((float) scenePoints.get(3).x, (float) scenePoints.get(3).y);
            path.close();

            /**
             * Using RectF to draw a fixed rectangle bounding box.
             */
            float[] xValues = {(float) scenePoints.get(0).x,
                    (float) scenePoints.get(1).x,
                    (float) scenePoints.get(2).x,
                    (float) scenePoints.get(3).x};
            float[] yValues = {(float) scenePoints.get(0).y,
                    (float) scenePoints.get(1).y,
                    (float) scenePoints.get(2).y,
                    (float) scenePoints.get(3).y};
            Arrays.sort(xValues);
            Arrays.sort(yValues);
            location.set(xValues[0], yValues[0], xValues[3], yValues[3]);
        } else return null;

        return new Recognition("",Pair.create(path, location));
    }

}
