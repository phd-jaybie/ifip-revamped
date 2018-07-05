package org.tensorflow.ifip.network;

/**
 * Created by deg032 on 1/2/18.
 */

import android.graphics.Bitmap;

import org.tensorflow.ifip.Classifier;
import org.tensorflow.ifip.env.Logger;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedList;
import java.util.List;

import javax.net.ssl.HttpsURLConnection;

public class RemoteDetector implements Classifier {

    public static final String TAG = "RemoteDetector";
    private static final Logger LOGGER = new Logger();

    private static String urlString;

    // This is the network fragment that handles all network activity/functions.
    private static NetworkFragment mNetworkFragment;
    private static String operatingMode = "SIFT";

    public static RemoteDetector create(String remoteUrl, String detectionMode) {
        final RemoteDetector detector = new RemoteDetector();

        urlString = "http://" + remoteUrl + ":8081"; // for user-input URL
        operatingMode = detectionMode;
        if (remoteUrl.isEmpty()) urlString = "http://150.229.118.255:8081"; // co-located edge server
        //urlString = "http://13.211.118.53:8081"; // remote cloud server

        return detector;
    }

    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        final byte[] byteArray = stream.toByteArray();

        int width = bitmap.getWidth();
        int height = bitmap.getHeight();

        InputStream inputStream;
        String result = null;

        try {
            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            long time = System.currentTimeMillis();

            conn.setDoOutput(true);
            conn.setDoInput(true);
            conn.setChunkedStreamingMode(0);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Accept-Encoding", "identity");
            conn.setRequestProperty("Connection", "Keep-Alive");
            conn.setRequestProperty("Content-type", "image/jpeg");
            conn.setRequestProperty("Detection-mode", operatingMode);
            conn.addRequestProperty("Content-length", byteArray.length + "");

            //conn.connect();

            OutputStream outputBuff = new BufferedOutputStream(conn.getOutputStream());

            LOGGER.d("Uploading image for remote processing.");

            try {
                //showToast("Uploaded to: " + mURL.toString());
                outputBuff.write(byteArray);
                //Log.d(TAG, "Uploaded.");
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                outputBuff.close();
            }

            try {
                int responseCode = conn.getResponseCode();
                if (responseCode != HttpsURLConnection.HTTP_OK) {
                    throw new IOException("HTTP error code: " + responseCode);
                }
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }

            try {
                String responseLength = conn.getHeaderField("Content-length");
                String responseType = conn.getContentType();
                inputStream = conn.getInputStream();
                if (responseType.equals("text/xml")) {
                    result = readStream(inputStream, responseLength);
                } else {
                    LOGGER.d("Error detection.");
                    return null;
                }
                LOGGER.d("Received from remote: " + result);
            } catch (IOException e) {
                e.printStackTrace();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return resultToList(result, height, width);
    }

    private List<Recognition> resultToList(String result, int height, int width){
        List<Recognition> results = new LinkedList<>();
        XmlOperator xmlParser = new XmlOperator();
        try {
            InputStream in = new ByteArrayInputStream(result.getBytes(StandardCharsets.UTF_8));
            results = xmlParser.parse(in, height, width);
        } catch (Exception e){
            e.printStackTrace();
        }
        return results;
    }

    private String readStream(InputStream stream, String length) throws IOException {
        int maxLength = Integer.valueOf(length);
        String result = null;

        // Read InputStream using the UTF-8 charset.
        InputStreamReader reader = new InputStreamReader(stream, "UTF-8");

        // Create temporary buffer to hold Stream data with specified max length.
        char[] buffer = new char[maxLength];
        // Populate temporary buffer with Stream data.
        int numChars = 0;
        int readSize = 0;
        while (numChars < maxLength && readSize != -1) {
            numChars += readSize;
            readSize = reader.read(buffer, numChars, buffer.length - numChars);
        }
        if (numChars != -1) {
            // The stream was not empty.
            // Create String that is actual length of response body if actual length was less than
            // max length.
            numChars = Math.min(numChars, maxLength);
            result = new String(buffer, 0, numChars);
        }
        //String[] results = result.split("\n");
        return result;

    }


    @Override
    public void enableStatLogging(final boolean logStats) {
        // do nothing
    }

    @Override
    public String getStatString() {
        return null;
    }

    @Override
    public void close() {
        // do nothing
    }
}
