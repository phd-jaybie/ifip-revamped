package org.tensorflow.ifip.phd.detector.cv;

/**
 * Created by deg032 on 1/2/18.
 */

import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.utils.Converters;
import org.tensorflow.ifip.env.Logger;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Vector;

import static org.tensorflow.ifip.MrCameraActivity.MIN_MATCH_COUNT;

public class HammingDetector implements MarkerDetector {

    private static final Logger LOGGER = new Logger();

    // This is a separate imageDetector which extracts Hamming markers.

    private static MatOfPoint canonicalMarkerCoords = new MatOfPoint();

    private static Size perspectiveSize;

    final Integer[][] BORDER_COORDINATES = {
            {0, 0},{0,1},{0,2},{0,3},{0,4},{0,5},{0,6},{1,0},{1,6},{2,0},{2,6},{3,0},
            {3,6},{4,0},{4,6},{5,0},{5,6},{6,0},{6,1},{6,2},{6,3},{6,4},{6,5},{6,6}
    };

    final Integer[][] HAMMINGCODE_MARKER_POSITIONS = {
            {1, 2}, {1, 3}, {1, 4},
            {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5},
            {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5},
            {4, 1}, {4, 2}, {4, 3}, {4, 4}, {4, 5},
            {5, 2}, {5, 3}, {5, 4}
    };

    final Integer[][] ORIENTATION_MARKER_COORDINATES = {
            {1, 1}, {1, 5}, {5, 1}, {5, 5}
    };

    public static HammingDetector create(){

        double warpedSize = 49;

        perspectiveSize = new Size(warpedSize, warpedSize);

        // python: canonical_marker_coords = np.array(((0, 0),
        //        (warped_size - 1, 0),
        //        (warped_size - 1, warped_size - 1),
        //        (0, warped_size - 1)),
        //        dtype='float32')
        List<Point> points = new ArrayList<>();
        points.add(new Point(0,0));
        points.add(new Point(warpedSize - 1, 0));
        points.add(new Point(warpedSize - 1, warpedSize - 1));
        points.add(new Point(0, warpedSize - 1));

        canonicalMarkerCoords.fromList(points);

        //MatOfInt cI = new MatOfInt(canonicalMarkerCoords);
        LOGGER.i("CanonicalMarkerCoords: %s",canonicalMarkerCoords.toList().toString());

        final HammingDetector d = new HammingDetector();
        return d;
    }

    @Override
    public List<Marker> detectMarkers(Bitmap bitmap){

        //Bitmap resized = Bitmap.createScaledBitmap(bitmap,1280,720,true);

        List<Marker> markers = new ArrayList<>();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        //LOGGER.d("Processed bitmap: %dx%d", height, width);

        Mat frame = new Mat();
        Utils.bitmapToMat(bitmap, frame);
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        // python: edges = cv2.Canny(gray, 10, 100)
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 100, 200,3,false);

        // python: contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        double minContourLength = (Math.min(width, height)) / 50.0;

        List<MatOfPoint> goodContours = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            //MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            //double contourLength = Imgproc.arcLength(curve, true);
            //final MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());

            if (Imgproc.contourArea(contour) >=  2*minContourLength*minContourLength) {
                goodContours.add(contour); // len(contour) --> contour.toList().size() //&& contourLength<=maxContourLength)
            }

        } //getting long enough contours

        LOGGER.i("Number of good contours: %d (%d)", goodContours.size(), contours.size());

        int counter = 0;
        for (final MatOfPoint contour : goodContours) {

            MatOfPoint2f sortedCurve;

            // python: approx_curve = cv2.approxPolyDP(contour, len(contour) * 0.01, True)
            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            double contourLength = Imgproc.arcLength(curve, true);
            //LOGGER.i("ContourLength = %f (%f)", contourLength, minContourLength);
            MatOfPoint2f approxCurve = new MatOfPoint2f();
            double epsilon = 0.1*contour.toList().size(); //len(contour) --> contour.toList().size() OR contourLength
            Imgproc.approxPolyDP(curve, approxCurve, epsilon, true);

            // python: if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve)): continue
            MatOfPoint approxCurvef1 = new MatOfPoint();
            approxCurve.convertTo(approxCurvef1, CvType.CV_32S);

            if ((approxCurve.toList().size() != 4) || (!Imgproc.isContourConvex(approxCurvef1))) continue;

            counter ++;

            String markerId = "Contour#: " + counter;

            try {

                //double approxContourLength = Imgproc.arcLength(approxCurve, true);
                //LOGGER.i("Epsilon: %f", epsilon);
                //LOGGER.i("Contour: %f, %s", contourLength, contour.toList().toString());
                //LOGGER.i("ApproxCurve: %f, %s", approxContourLength, approxCurve.toList().toString());

                // python: sorted_curve = array(cv2.convexHull(approx_curve, clockwise=False), dtype='float32')
                sortedCurve = sortCurve(approxCurvef1);

                // python: persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
                Mat sortedCurveM = Converters.vector_Point2f_to_Mat(sortedCurve.toList());
                Mat canonicalM = Converters.vector_Point2f_to_Mat(canonicalMarkerCoords.toList());
                Mat perspectiveTransform = Imgproc.getPerspectiveTransform(sortedCurveM, canonicalM);

                // python: warped_img = cv2.warpPerspective(img, persp_transf, (warped_size, warped_size))
                Mat warpedImage = new Mat();

                Imgproc.warpPerspective(frame, warpedImage, perspectiveTransform, perspectiveSize);

                //python: warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
                Mat warpedGray = new Mat();
                Imgproc.cvtColor(warpedImage, warpedGray, Imgproc.COLOR_BGR2GRAY);

                // warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
                Mat warpedBin = new Mat();
                Imgproc.threshold(warpedGray, warpedBin, 127, 255, Imgproc.THRESH_BINARY);

                Boolean[][] marker = new Boolean[7][7];

                // python: marker = marker.mean(axis=3).mean(axis=1)
                // python: marker[marker < 127] = 0
                // python: marker[marker >= 127] = 1
                for (int row = 0; row < 7; row++) {
                    for (int col = 0; col < 7; col++) {
                        Mat subMat = warpedBin.submat(row * 7, row * 7 + 7, col * 7, col * 7 + 7);

                        //getting mean at axis 3: as if *.mean(axis=3)
                        double[] meanSub3 = new double[7];
                        for (int i = 0; i < 7; i++) {
                            double axis3Sum = 0;

                            for (int d = 0; d < 7; d++) axis3Sum += subMat.get(i, d)[0];

                            meanSub3[i] = axis3Sum / 7.0;
                        }

                        //getting mean at axis 1: as if *.mean(axis=1)
                        int axis1Sum = 0;
                        for (double mean3 : meanSub3) axis1Sum += mean3;

                        if (axis1Sum / 7.0 < 127)
                            marker[row][col] = false; // as if python: marker[marker < 127] = 0
                        else
                            marker[row][col] = true; // as if python: marker[marker >= 127] = 1
                    }
                }

                //            if (Imgproc.isContourConvex(approxCurvef1))
                //                markerId = "Convex: " + approxCurve.toList().size() + " corners";
                //            else
                //                markerId = "Non-Convex: " + approxCurve.toList().size() + " corners";

                  try {
                    // python: marker = validate_and_turn(marker)
                    Boolean[][] validatedMarker = validateAndTurn(marker);

                    // python: hamming_code = extract_hamming_code(marker)
                    final BitSet hammingCode = extractHammingCode(validatedMarker);

                    // python: marker_id = int(decode(hamming_code), 2)
                    //final int markerId = decode(hammingCode);
                      markerId = hammingCode.toString();

                      LOGGER.i("Contour#: %d, MarkerId: %s", counter, markerId);
                    //python: markers_list.append(HammingMarker(id=marker_id, contours=approx_curve))
                    //markers.add(new Marker(markerId, approxCurve));

                } catch (Exception e) {
                    e.printStackTrace();
                }

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                markers.add(new Marker(markerId, approxCurve));
            }
        }

        return markers;
    }

    private MatOfPoint2f sortCurve(MatOfPoint curve) {

        //calculate the center of mass of our contour image using moments
        Moments moment = Imgproc.moments(curve);
        int mx = (int) (moment.get_m10() / moment.get_m00());
        int my = (int) (moment.get_m01() / moment.get_m00());

        LOGGER.i("Moment: {%d, %d}", mx, my);

        //SORT POINTS RELATIVE TO CENTER OF MASS
        Point[] sortedPoints = new Point[4];
        List<Point> curvePoints = new ArrayList<>();

        curvePoints = curve.toList();

        for(final Point point: curvePoints){

            double datax = point.x;
            double datay = point.y;

            if(datax < mx && datay < my){
                sortedPoints[0]=new Point(datax,datay);
            }else if(datax > mx && datay < my){
                sortedPoints[1]=new Point(datax,datay);
            }else if (datax > mx && datay > my){
                sortedPoints[2]=new Point(datax,datay);
            }else if (datax < mx && datay > my){
                sortedPoints[3]=new Point(datax,datay);
            }
        }

        return new MatOfPoint2f(sortedPoints);

    }

    private Boolean[][] validateAndTurn(Boolean[][] marker) throws Exception{

        Boolean[][] validatedMarker = new Boolean[7][7];

        //# first, lets make sure that the border contains only zeros
        // python: for crd in BORDER_COORDINATES:
        // python: if marker[crd[0], crd[1]] != 0.0:
        // python: raise ValueError('Border contians not entirely black parts.')
        for (Integer[] borderPos: BORDER_COORDINATES) {
            if (!marker[borderPos[0]][borderPos[1]]) {
                throw new Exception("Border contains not entirely black parts.");
            }
        }

        //# search for the corner marker for orientation and make sure, there is only 1
        // python: orientation_marker = None
        // for crd in ORIENTATION_MARKER_COORDINATES:
        Integer[] orientationMarker = new Integer[2];

        for (Integer[] orientationPos: ORIENTATION_MARKER_COORDINATES) {
            //marker_found = False
            boolean markerFound = false;

            //if marker[crd[0], crd[1]] == 1.0:
            //marker_found = True
            if (marker[orientationPos[0]][orientationPos[1]]) markerFound = true;

            //if marker_found and orientation_marker:
            if (markerFound && orientationMarker[0]!=null) {
                //raise ValueError('More than 1 orientation_marker found.')
g                throw new Exception("More than 1 orientation_marker found.");
            } else if (markerFound) { //elif marker_found:
                //orientation_marker = crd
                orientationMarker = orientationPos;
            }
        }

        //if not orientation_marker:
        if (orientationMarker[0] == null) {
            //raise ValueError('No orientation marker found.')
            throw  new Exception("No orientation marker found.");
        }

        int rotation = 0;

        if (orientationMarker[0] == 1 && orientationMarker[1] ==5) rotation = 1;
        //elif orientation_marker == [5, 5]:
        //rotation = 2
        else if (orientationMarker[0] == 5 && orientationMarker[1] ==5) rotation = 2;
        //elif orientation_marker == [5, 1]:
        //rotation = 3
        else if (orientationMarker[0] == 5 && orientationMarker[1] ==1) rotation = 3;

        //marker = rot90(marker, k=rotation)
        validatedMarker = rotateN(marker, rotation);

        //return marker
        LOGGER.d("Successful validate marker.");
        return validatedMarker;
    }

    private static Boolean[][] rotateN(Boolean[][] array, int rotationN){

        while(rotationN>0){
            Boolean[][] newarray = new Boolean[7][7];

            for (int i = 0; i < 7; i++){
                for (int j =0; j<7; j++){
                    newarray[i][j] = array [j][7-1-i];
                }
            }

            array = newarray;
            rotationN--;
        }

        return array;
    }

    private BitSet extractHammingCode(Boolean[][] marker) {
        BitSet hammingCode = new BitSet();

        // python: hamming_code += str(int(mat[pos[0], pos[1]]))
        int bitPos = 0;
        for (Integer[] pos: HAMMINGCODE_MARKER_POSITIONS){
            if (marker[pos[0]][pos[1]]) hammingCode.set(bitPos, true);
            else hammingCode.set(bitPos, false);
        }

        return hammingCode;
    }

    private Integer decode(BitSet hammingCode) throws Exception {

        //decoded_code = ''
        int code = 0;

        //if len(bits) % 7 != 0:
        if (hammingCode.length()%7 !=0) {
            //raise ValueError('Only a multiple of 7 as bits are allowed.')
            LOGGER.d("Hamming Code: %s", hammingCode.toString());
            throw new Exception("Only a multiple of 7 as bits are allowed.");
        }

        //seven_bits = bits[:7]
        //uncorrected_bit_array = generate_bit_array(seven_bits)
        //corrected_bit_array = parity_correct(uncorrected_bit_array)
        //decoded_bits = matrix_array_multiply_and_format(REGENERATOR_MATRIX, corrected_bit_array)
        //decoded_code += ''.join(decoded_bits)
        //bits = bits[7:]

        //return decoded_code

        return code;
    }

}
