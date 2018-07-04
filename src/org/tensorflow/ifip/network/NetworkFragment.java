package org.tensorflow.ifip.network;

import android.content.res.AssetManager;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.content.Context;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.Nullable;


import org.tensorflow.ifip.env.Logger;
import org.tensorflow.ifip.phd.MrObjectManager;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;

import javax.net.ssl.HttpsURLConnection;

/**
 * Created by deg032 on 22/2/18.
 */

public class NetworkFragment extends Fragment {

    public static final String TAG = "NetworkFragment";
    private static final Logger LOGGER = new Logger();


    private static final String URL_KEY = "UrlKey";

    private NetworkListener networkListener;
    private NetworkTask mNetworkTask;
    private String mUrlString;

    private NetworkServer networkServer;
    private String mXmlPayload;
    private byte[] mBytesPayload;

    private enum NetworkMode {
        XML_SHARE, REMOTE_PROCESSOR;
    }

    private static NetworkMode MODE;

    /**
     * Static initializer for NetworkFragment that sets the URL of the host it will be downloading
     * from.
     */
    public static NetworkFragment getInstance(FragmentManager fragmentManager, String url) {
        // Recover NetworkFragment in case we are re-creating the Activity due to a config change.
        // This is necessary because NetworkFragment might have a task that began running before
        // the config change and has not finished yet.
        // The NetworkFragment is recoverable via this method because it calls
        // setRetainInstance(true) upon creation.

        NetworkFragment networkFragment = (NetworkFragment) fragmentManager
                .findFragmentByTag(NetworkFragment.TAG);
        if (networkFragment == null) {
            networkFragment = new NetworkFragment();
            Bundle args = new Bundle();
            args.putString(URL_KEY, url);
            networkFragment.setArguments(args);
            fragmentManager.beginTransaction().add(networkFragment, TAG).commit();
        }

        return networkFragment;
    }

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Retain this Fragment across configuration changes in the host Activity.
        setRetainInstance(true);
        mUrlString = getArguments().getString(URL_KEY);
    }

    @Override
    public void onAttach(Context context) {
        super.onAttach(context);
        if (networkServer!= null) networkServer.start();
        // Host Activity will handle callbacks from task.
        networkListener = (NetworkListener) context;
    }

    @Override
    public void onDetach() {
        super.onDetach();
        // Clear reference to host Activity.
        if (networkServer!= null) networkServer.stop();
        networkListener = null;
    }

    @Override
    public void onDestroy() {
        // Cancel task when Fragment is destroyed.
        cancelDownload();
        if (networkServer!= null) networkServer.stop();
        super.onDestroy();
    }

    /**
     * Start non-blocking execution of NetworkTask.
     */
    public void startNetwork(String mUrl) {
        cancelDownload();
        mNetworkTask = new NetworkTask();
        mNetworkTask.execute(mUrl);
    }

    /**
     * Shares the stored objects in this users 'live' object space.
     * */
    public void shareObjects(String Xmlpayload){
        // Convert the list of public objects to an XML. Then send out.
        mXmlPayload = Xmlpayload;
        MODE = NetworkMode.XML_SHARE;

        // Instead of using a single URL, a list of URLs can be iterated and used.
        // For the meantime, we use a single URL.
        startNetwork(mUrlString);
    }

    /**
     * This one uploads the raw image to a server for processing. And receives the object locations.
     *
     * Currently, I am using different class (i.e. RemotedEtector) to do this, thus, this is not
     * actually used. Will have to reconcile this NetworkFragment library with that one.
     */
    public void remoteImageProcessing(byte[] payload){
        // Convert the list of public objects to an XML. Then send out.
        mBytesPayload = payload;
        MODE = NetworkMode.REMOTE_PROCESSOR;
        // Instead of using a single URL, a list of URLs can be iterated and used.
        // For the meantime, we use a single URL.
        startNetwork(mUrlString);
    }

    /**
     * Cancel (and interrupt if necessary) any ongoing DownloadTask execution.
     */
    public void cancelDownload() {
        if (mNetworkTask != null) {
            mNetworkTask.cancel(true);
            mNetworkTask = null;
        }
    }

    /**
     * For listening to a port, we create a simple server that will do that.
     */

    public void startServer(int mPort, AssetManager assets){
        networkServer = new NetworkServer(mPort, assets);
    }

    public void setServerListener(NetworkListener networkListener){
        if (networkServer != null) networkServer.setNetworkListener(networkListener);
    }

    public List<MrObjectManager.MrObject> getObjects(){
        // parse the XML file received and extract the objects.
        return null;
    }


    /**
     * Implementation of AsyncTask that runs a network operation on a background thread.
     */
    private class NetworkTask extends AsyncTask<String, Void, String> {

        @Override
        protected String doInBackground(String... params) {
            try {
                if (MODE == NetworkMode.XML_SHARE) return shareData(params[0]);
                else return processImage(params[0]);
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        }

        @Override
        protected void onPostExecute(String result) {
            try {
                networkListener.uploadComplete();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private String shareData(String urlString) throws IOException {
            final String payload = mXmlPayload;

            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            /*try {
                URL url = new URL(urlString);
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.connect();
                is = conn.getInputStream();
                return convertToString(is);
            } finally {
                if (is != null) {
                    is.close();
                }
            }*/

            String[] result;

            try {
                InputStream inputStream = null;

                long time = System.currentTimeMillis();

                conn.setDoOutput(true);
                conn.setDoInput(true);
                conn.setChunkedStreamingMode(0);
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Accept-Encoding", "identity");
                conn.setRequestProperty("Connection", "Keep-Alive");
                conn.setRequestProperty("Content-type", "xml");
                conn.addRequestProperty("Content-length", payload.getBytes().length + "");

                //conn.connect();

                OutputStream outputBuff = new BufferedOutputStream(conn.getOutputStream());

                LOGGER.d("Sharing objects to remote.");

                try {
                    //showToast("Uploaded to: " + mURL.toString());
                    outputBuff.write(payload.getBytes());
                    //Log.d(TAG, "Uploaded.");
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    outputBuff.close();
                }

                int responseCode = conn.getResponseCode();
                String responseLength = conn.getHeaderField("Content-length");

                if (responseCode != HttpsURLConnection.HTTP_OK) {
                    throw new IOException("HTTP error code: " + responseCode);
                }

                try {
                    inputStream = conn.getInputStream();
                    // Extract the input stream.
                    //showToast("Received from remote: " + result);
                } catch (IOException e) {
                    e.printStackTrace();
                }

            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                conn.disconnect();
            }

            return "Success";
        }


        private String processImage(String urlString) throws IOException {
            final byte[] payload = mBytesPayload;

            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();

            String result;

            try {
                InputStream inputStream = null;

                long time = System.currentTimeMillis();

                conn.setDoOutput(true);
                conn.setDoInput(true);
                conn.setChunkedStreamingMode(0);
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Accept-Encoding", "identity");
                conn.setRequestProperty("Connection", "Keep-Alive");
                conn.setRequestProperty("Content-type", "image/jpeg");
                conn.addRequestProperty("Content-length", payload.length + "");

                //conn.connect();

                OutputStream outputBuff = new BufferedOutputStream(conn.getOutputStream());

                LOGGER.d("Uploading image for remote processing.");

                try {
                    //showToast("Uploaded to: " + mURL.toString());
                    outputBuff.write(payload);
                    //Log.d(TAG, "Uploaded.");
                } catch (IOException e) {
                    e.printStackTrace();
                } finally {
                    outputBuff.close();
                }

                int responseCode = conn.getResponseCode();
                String responseLength = conn.getHeaderField("Content-length");

                if (responseCode != HttpsURLConnection.HTTP_OK) {
                    throw new IOException("HTTP error code: " + responseCode);
                }

                try {
                    inputStream = conn.getInputStream();
                    result = readStream(inputStream, responseLength);
                    LOGGER.v("Received from remote: " + result);
                } catch (IOException e) {
                    e.printStackTrace();
                }

            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                conn.disconnect();
            }

            return "Success";
        }

        private String convertToString(InputStream is) throws IOException {
            BufferedReader r = new BufferedReader(new InputStreamReader(is));
            StringBuilder total = new StringBuilder();
            String line;
            while ((line = r.readLine()) != null) {
                total.append(line);
            }
            return new String(total);
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
    }

}
