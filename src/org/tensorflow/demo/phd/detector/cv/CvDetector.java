package org.tensorflow.demo.phd.detector.cv;

/**
 * Created by deg032 on 1/2/18.
 */

import android.graphics.Bitmap;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.Pair;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.tensorflow.demo.simulator.AppRandomizer;

public interface CvDetector{

    class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        //private final String id;

        /**
         * Display name for the recognition.
         */
        private String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        //private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private Pair<Path, RectF> location;

        public Recognition(
                final String title, final Pair<Path, RectF> location) {
            this.title = title;
            this.location = location;
        }

        public Recognition(){

        }

        public String getTitle() {
            return title;
        }

        public Pair<Path, RectF> getLocation() {
            return location;
        }

        public void setTitle(String title) {
            this.title = title;
        }

        public void setLocation(Pair<Path, RectF> location) {
            this.location = location;
        }

    }

    class QueryImage {
        private Mat QryImageMat;
        private MatOfKeyPoint QryKeyPoints;
        private Mat QryDescriptors;

        public QueryImage(Mat qryImageMat, MatOfKeyPoint qryKeyPoints, Mat qryDescriptors){
            this.QryImageMat = qryImageMat;
            this.QryDescriptors = qryDescriptors;
            this.QryKeyPoints = qryKeyPoints;
        }

        public QueryImage(){
        }

        public void setQryDescriptors(Mat qryDescriptors) {
            this.QryDescriptors = qryDescriptors;
        }

        public Mat getQryDescriptors() {
            return QryDescriptors;
        }

        public void setQryImageMat(Mat qryImageMat) {
            this.QryImageMat = qryImageMat;
        }

        public Mat getQryImageMat() {
            return QryImageMat;
        }

        public void setQryKeyPoints(MatOfKeyPoint qryKeyPoints) {
            this.QryKeyPoints = qryKeyPoints;
        }

        public MatOfKeyPoint getQryKeyPoints() {
            return QryKeyPoints;
        }

    }

    QueryImage imageDetector(Bitmap bitmap);

    Recognition imageDetector(Bitmap bitmap, AppRandomizer.ReferenceImage reference);

    Recognition getTransformation(QueryImage queryImage, AppRandomizer.ReferenceImage reference);

}
