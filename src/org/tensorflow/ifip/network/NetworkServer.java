package org.tensorflow.ifip.network;

import android.content.res.AssetManager;
import android.text.TextUtils;
import android.util.Log;

import org.tensorflow.ifip.env.Logger;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;

/**
 * Created by deg032 on 22/2/18.
 */

public class NetworkServer implements Runnable {

    private static final String TAG = "SimpleWebServer";
    private static final Logger LOGGER = new Logger();

    private static NetworkListener networkListener;
    /**
     * The port number we listen to
     */
    private final int mPort;

    /**
     * {@link android.content.res.AssetManager} for loading files to serve.
     */
    private final AssetManager mAssets;

    /**
     * True if the server is running.
     */
    private boolean mIsRunning;

    /**
     * The {@link java.net.ServerSocket} that we listen to.
     */
    private ServerSocket mServerSocket;

    /**
     * WebServer constructor.
     */
    public NetworkServer(int port, AssetManager assets) {
        mPort = port;
        mAssets = assets;
    }

    public void setNetworkListener(NetworkListener mNetworkListener){
        networkListener = mNetworkListener;
    }

    /**
     * This method starts the web server listening to the specified port.
     */
    public void start() {
        mIsRunning = true;
        new Thread(this).start();
        LOGGER.d("Server started.");
    }

    /**
     * This method stops the web server
     */
    public void stop() {
        try {
            mIsRunning = false;
            if (null != mServerSocket) {
                mServerSocket.close();
                mServerSocket = null;
                LOGGER.d("Server stopped.");
            }
        } catch (IOException e) {
            LOGGER.e(TAG, "Error closing the server socket.", e);
        }
    }

    public int getPort() {
        return mPort;
    }

    @Override
    public void run() {
        try {
            mServerSocket = new ServerSocket(mPort);
            while (mIsRunning) {
                Socket socket = mServerSocket.accept();
                handle(socket);
                socket.close();
            }
        } catch (SocketException e) {
            // The server was stopped; ignore.
        } catch (IOException e) {
            Log.e(TAG, "Web server error.", e);
        }
    }

    /**
     * Respond to a request from a client.
     *
     * @param socket The client socket.
     * @throws IOException
     */
    private void handle(Socket socket) throws IOException {
        XmlOperator xmlParser = new XmlOperator();
        BufferedReader reader = null;
        PrintStream output = new PrintStream(socket.getOutputStream());

        try {

            // Read HTTP headers and parse out the route.
            reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            if (reader== null) return;

            String line = reader.readLine();

            if (!line.startsWith("POST")) return;

            int contentLength = 0;

            while (!TextUtils.isEmpty(line = reader.readLine())) {
                LOGGER.i(line);
                if (line.startsWith("Content-Length:")) {
                    int start = line.indexOf(':') + 2;
                    contentLength = Integer.valueOf(line.substring(start, line.length()));
                }
            }

            StringBuilder body = new StringBuilder();
            while (body.length() < contentLength) {
                body.append((char) reader.read());
            }

            // LOGGER.i("Body: " + body.toString());
            InputStream is = new ByteArrayInputStream(body.toString().getBytes(StandardCharsets.UTF_8));

            BufferedReader bodyReader = new BufferedReader(new InputStreamReader(is));
            String bodyLine = null;
            while (!TextUtils.isEmpty(bodyLine = bodyReader.readLine())) LOGGER.i("Body:" + bodyLine);

            /*List fromNetwork = new ArrayList();

            try {
                fromNetwork = xmlParser.parse(is);
                networkListener.receivedFromNetwork(fromNetwork);
                networkListener.setReceiveFlag(true);

            } catch (Exception e) {
                e.printStackTrace();
                networkListener.setReceiveFlag(false);
                writeServerError(output);
                return;
            }*/

            // Send out the content.
            output.println("HTTP/1.0 200 OK");
            /*output.println("Content-Type: " + detectMimeType(route));
            output.println("Content-Length: " + bytes.length);
            output.println();
            output.write(bytes);*/
            output.flush();
        } finally {
            if (null != output) {
                output.close();
            }
            if (null != reader) {
                reader.close();
            }
        }
    }

    /**
     * Writes a server error response (HTTP/1.0 500) to the given output stream.
     *
     * @param output The output stream.
     */
    private void writeServerError(PrintStream output) {
        output.println("HTTP/1.0 500 Internal Server Error");
        output.flush();
    }

    /**
     * Loads all the content of {@code fileName}.
     *
     * @param fileName The name of the file.
     * @return The content of the file.
     * @throws IOException
     */
    private byte[] loadContent(String fileName) throws IOException {
        InputStream input = null;
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            input = mAssets.open(fileName);
            byte[] buffer = new byte[1024];
            int size;
            while (-1 != (size = input.read(buffer))) {
                output.write(buffer, 0, size);
            }
            output.flush();
            return output.toByteArray();
        } catch (FileNotFoundException e) {
            return null;
        } finally {
            if (null != input) {
                input.close();
            }
        }
    }

    /**
     * Detects the MIME type from the {@code fileName}.
     *
     * @param fileName The name of the file.
     * @return A MIME type.
     */
    private String detectMimeType(String fileName) {
        if (TextUtils.isEmpty(fileName)) {
            return null;
        } else if (fileName.endsWith(".html")) {
            return "text/html";
        } else if (fileName.endsWith(".js")) {
            return "application/javascript";
        } else if (fileName.endsWith(".css")) {
            return "text/css";
        } else {
            return "application/octet-stream";
        }
    }

}
