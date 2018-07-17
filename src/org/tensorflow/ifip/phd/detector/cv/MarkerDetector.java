package org.tensorflow.ifip.phd.detector.cv;

/**
 * Created by deg032 on 1/2/18.
 */

import android.graphics.Bitmap;
import android.graphics.Path;
import android.graphics.RectF;

import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;

import java.util.Arrays;
import java.util.List;

public interface MarkerDetector {

    class Marker {

        private String markerId;

        private MatOfPoint2f contour;

        public Marker(final String markerId, MatOfPoint2f contour) {
            this.markerId = markerId;
            this.contour = contour;
        }

        public String getId() {
            return markerId;
        }

        public MatOfPoint2f getContour() {
            return contour;
        }

/*
        public void setMarkerId(String markerId) {
            this.markerId = markerId;
        }

        public void setContour(MatOfPoint2f contour) {
            this.contour = contour;
        }
*/

        public Path contourToPath() {
            Path path = new Path();

            List<Point> contourPoints = this.contour.toList();

            path.moveTo((float) contourPoints.get(0).x, (float) contourPoints.get(0).y);
            path.lineTo((float) contourPoints.get(1).x, (float) contourPoints.get(1).y);
            path.lineTo((float) contourPoints.get(2).x, (float) contourPoints.get(2).y);
            path.lineTo((float) contourPoints.get(3).x, (float) contourPoints.get(3).y);
            path.close();

            return path;
        }

        public RectF contourToRect() {
            RectF location = new RectF();

            List<Point> contourPoints = this.contour.toList();
            float[] xValues = {(float) contourPoints.get(0).x,
                    (float) contourPoints.get(1).x,
                    (float) contourPoints.get(2).x,
                    (float) contourPoints.get(3).x};
            float[] yValues = {(float) contourPoints.get(0).y,
                    (float) contourPoints.get(1).y,
                    (float) contourPoints.get(2).y,
                    (float) contourPoints.get(3).y};
            Arrays.sort(xValues);
            Arrays.sort(yValues);
            location.set(xValues[0], yValues[0], xValues[3], yValues[3]);

            return location;
        }

    }

    List<Marker> detectMarkers(Bitmap bitmap);

}
