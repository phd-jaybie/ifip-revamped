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
import org.tensorflow.ifip.env.Logger;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Vector;

public class HammingDetector implements MarkerDetector {

    private static final Logger LOGGER = new Logger();

    // This is a separate imageDetector which extracts Hamming markers.

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
        final HammingDetector d = new HammingDetector();
        return d;
    }

    @Override
    public List<Marker> detectMarkers(Bitmap bitmap){

        List<Marker> markers = new ArrayList<>();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        LOGGER.d("Input bitmap: %dx%d", width, height);

        Mat frame = new Mat();
        Utils.bitmapToMat(bitmap, frame);
        Mat gray = new Mat();
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 10, 100);

        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        double minContourLength = (Math.min(width, height)) / 50.0;
        double maxContourLength = 2*(width + height);

        List<MatOfPoint> goodContours = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            double contourLength = Imgproc.arcLength(curve, true);
            if (contourLength > minContourLength) //&& contourLength<=maxContourLength)
                goodContours.add(contour); // len(contour) --> contour.toList().size()
        } //getting long enough contours

        double warpedSize = 49;

        //canonical_marker_coords = np.array(((0, 0),
        //        (warped_size - 1, 0),
        //        (warped_size - 1, warped_size - 1),
        //        (0, warped_size - 1)),
        //        dtype='float32')
        List<Point> points = new ArrayList<>();
        points.add(new Point(0,0));
        points.add(new Point(warpedSize - 1, 0));
        points.add(new Point(warpedSize - 1, warpedSize - 1));
        points.add(new Point(0, warpedSize - 1));

        MatOfPoint canonicalMarkerCoords = new MatOfPoint();
        canonicalMarkerCoords.fromList(points);

//        Mat canonicalMarkerCoords = new Mat(4,1, CvType.CV_32FC2);
//        canonicalMarkerCoords.put(0, 0, new double[]{0, 0});
//        canonicalMarkerCoords.put(1, 0, new double[]{warpedSize - 1, 0});
//        canonicalMarkerCoords.put(2, 0, new double[]{warpedSize - 1, warpedSize - 1});
//        canonicalMarkerCoords.put(3, 0, new double[]{0, warpedSize - 1});

        //MatOfInt canonicalMarkerCoords = new MatOfInt(canonicalMarkerCoordsM);
        //LOGGER.i("Canonical coords %S", canonicalMarkerCoords.toList().toString());

        LOGGER.i("Number of good contours: %d (%d)", goodContours.size(), contours.size());

        //MatOfInt cI = new MatOfInt(canonicalMarkerCoords);
        //LOGGER.i("CanonicalMarkerCoords: %s", cI.toList().toString());

        for (final MatOfPoint contour : goodContours) {

            // python: approx_curve = cv2.approxPolyDP(contour, len(contour) * 0.01, True)
            MatOfPoint2f curve = new MatOfPoint2f(contour.toArray());
            double contourLength = Imgproc.arcLength(curve, true);

            //LOGGER.i("ContourLength = %f (%f)", contourLength, minContourLength);

            MatOfPoint2f approxCurve = new MatOfPoint2f();

            double epsilon = 0.01*contour.toList().size(); //len(contour) --> contour.toList().size() OR contourLength

            Imgproc.approxPolyDP(curve, approxCurve,
                    epsilon, true);

            // python: if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve)): continue
            // len(approxCurve) gets the number of elements of the matrix which should be always 4.
            //LOGGER.i("ApproxCurve: %s",approxCurve.toList().toString());
            //LOGGER.i("ApproxCurve (List) length: %d",approxCurve.toList().size());
            //LOGGER.i("ApproxCurve (Array) length: %d",approxCurve.toArray().length);

            if ((approxCurve.toArray().length > 20) ||
                    (approxCurve.toArray().length < 4) ||
                    (!Imgproc.isContourConvex(contour))) continue;

            double approxContourLength = Imgproc.arcLength(approxCurve, true);
            LOGGER.i("ApproxCurve: %f, %s", approxContourLength, approxCurve.toList().toString());

            // python: sorted_curve = array(cv2.convexHull(approx_curve, clockwise=False), dtype='float32')
            MatOfInt sortedCurve = new MatOfInt();
            MatOfPoint approxCurveF1 = new MatOfPoint();
            approxCurve.convertTo(approxCurveF1,CvType.CV_32S);

            Imgproc.convexHull(approxCurveF1, sortedCurve, false);
            LOGGER.i("Converted ApproxCurve: %s", approxCurveF1.toArray().toString());

            LOGGER.i("SortedCurve: [%dx%d] %s", sortedCurve.rows(),
                    sortedCurve.cols(),
                    sortedCurve.toList().toString());

            // python: persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
            Mat perspectiveTransform = Imgproc.getPerspectiveTransform(sortedCurve, canonicalMarkerCoords);

            // python: warped_img = cv2.warpPerspective(img, persp_transf, (warped_size, warped_size))
            Mat warpedImage = new Mat();
            Size perspectiveSize = new Size(warpedSize, warpedSize);
            Imgproc.warpPerspective(frame, warpedImage, perspectiveTransform, perspectiveSize);

            //python: warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            Mat warpedGray = new Mat();
            Imgproc.cvtColor(warpedImage, warpedGray, Imgproc.COLOR_BGR2GRAY);

            // warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
            Mat warpedBin = new Mat();
            Imgproc.threshold(warpedGray, warpedBin, 127, 255, Imgproc.THRESH_BINARY);

            // marker = warped_bin.reshape([int(MARKER_SIZE), int(warped_size / MARKER_SIZE), int(MARKER_SIZE), int(warped_size / MARKER_SIZE)])
            // The python (i.e. reshape) basically transforms warped_bin (of shape 49, 49)
            // into a 4D-array --> marker (7, 7, 7, 7)

            Boolean[][] marker = new Boolean[7][7];

            // python: marker = marker.mean(axis=3).mean(axis=1)
            // python: marker[marker < 127] = 0
            // python: marker[marker >= 127] = 1
            for (int row = 0; row < 7; row++) {
                for (int col = 0; col < 7; col++) {
                    Mat subMat = warpedBin.submat(row * 7, row * 7 + 7, col * 7, col * 7 + 7);

                    MatOfInt subMatInt = new MatOfInt(CvType.CV_32FC2);
                    subMat.convertTo(subMatInt, CvType.CV_32FC2);

                    //getting mean at axis 3: as if *.mean(axis=3)
                    MatOfInt meanSub3 = new MatOfInt();
                    for (int i = 0; i < 7; i++) {
                        int axis3Sum = 0;
                        for (int d : subMatInt.toArray()) axis3Sum += d;
                        meanSub3.put(0, i, axis3Sum / 7.0);
                    }

                    //getting mean at axis 1: as if *.mean(axis=1)
                    int axis1Sum = 0;
                    for (int d : meanSub3.toArray()) axis1Sum += d;

                    if (axis1Sum / 7.0 < 127)
                        marker[row][col] = false; // as if python: marker[marker < 127] = 0
                    else
                        marker[row][col] = true; // as if python: marker[marker >= 127] = 1
                }
            }

            try {
                // python: marker = validate_and_turn(marker)
                Boolean[][] validatedMarker = validateAndTurn(marker);

                // python: hamming_code = extract_hamming_code(marker)
                final BitSet hammingCode = extractHammingCode(validatedMarker);

                // python: marker_id = int(decode(hamming_code), 2)
                //final int markerId = decode(hammingCode);
                final String markerId = hammingCode.toString();

                //python: markers_list.append(HammingMarker(id=marker_id, contours=approx_curve))
                markers.add(new Marker(markerId, approxCurve));

            } catch (Exception e) {
                e.printStackTrace();
            }
        }


        return markers;
    }

    private Boolean[][] validateAndTurn(Boolean[][] marker) throws Exception{

        Boolean[][] validatedMarker = new Boolean[7][7];

        //# first, lets make sure that the border contains only zeros
        // python: for crd in BORDER_COORDINATES:
        // python: if marker[crd[0], crd[1]] != 0.0:
        // python: raise ValueError('Border contians not entirely black parts.')
        for (Integer[] borderPos: BORDER_COORDINATES) {
            if (marker[borderPos[0]][borderPos[1]]) {
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
            if (markerFound && !orientationMarker.toString().isEmpty()) {
                //raise ValueError('More than 1 orientation_marker found.')
                throw new Exception("More than 1 orientation_marker found.");
            } else if (markerFound) { //elif marker_found:
                //orientation_marker = crd
                orientationMarker = orientationPos;
            }
        }

        //if not orientation_marker:
        if (orientationMarker.toString().isEmpty()) {
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
