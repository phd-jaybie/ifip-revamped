/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.Manifest;
import android.app.Fragment;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.hardware.Camera;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.Image.Plane;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Build;
import android.os.Bundle;
import android.os.Trace;
import android.support.v4.app.FragmentActivity;
import android.util.Size;
import android.view.KeyEvent;
import android.view.Surface;
import android.view.WindowManager;
import android.widget.Toast;

import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.network.NetworkFragment;
import org.tensorflow.demo.network.NetworkListener;
import org.tensorflow.demo.network.XmlOperator;
import org.tensorflow.demo.phd.abstraction.MrObjectManager;
import org.tensorflow.demo.simulator.App;
import org.tensorflow.demo.simulator.SingletonAppList;
import org.tensorflow.demo.threading.ProcessManager;
import org.tensorflow.demo.threading.ProcessTask;

import java.io.FileWriter;
import java.nio.ByteBuffer;
import java.util.List;

public abstract class MrThreadedCameraActivity extends FragmentActivity
    implements OnImageAvailableListener, Camera.PreviewCallback, NetworkListener {

  //temporarily deactivated spinner

  protected static Integer utilityHit;
  protected static final String[] secretObjects = new String[]
          {"person", "bed", "toilet", "laptop", "cell phone"}; //high sensitivity objects

  private static final Logger LOGGER = new Logger();
  protected static FileWriter logWriter;
  protected static boolean receiveFlag = false;


  private static final int PERMISSIONS_REQUEST = 1;

  private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
  private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

  private boolean debug = false;

  protected static ProcessTask mProcessHandler;
  //private HandlerThread handlerThread;

  protected static final int CAPTURE_TIMEOUT = 19;
  protected static final int INSTANCE_TIMEOUT = 10;
  protected static long[] overallTimes;
  protected static long[] detectionTimes;

  private boolean useCamera2API;
  private boolean isProcessingFrame = false;
  private byte[][] yuvBytes = new byte[3][];
  private int[] rgbBytes = null;
  private int yRowStride;

  protected int previewWidth = 0;
  protected int previewHeight = 0;
  protected static int inputSize = 300; // default to default crop size

  private Runnable postInferenceCallback;
  private Runnable imageConverter;

  // network variables
  protected NetworkFragment mNetworkFragment;
  protected AssetManager mAssets;

  // This is used by the MrCameraActivity child with networking.
  protected static String NetworkMode;
  protected static String remoteUrl;
  protected static boolean fastDebug;

  // This is the Global object manager for MrObjects.
  protected static MrObjectManager manager;

  // Global values and containers for detection using OpenCV.
  public static Integer MIN_MATCH_COUNT = 30;

  protected static SingletonAppList singletonAppList = SingletonAppList.getInstance();

  protected static List<App> appList;
  protected static String appListText;

  protected static ProcessManager sProcessManager;

  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);
    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    getActionBar().setDisplayHomeAsUpEnabled(true);

    setContentView(R.layout.activity_camera);
    //startBackgroundThread();

    if (hasPermission()) {
      setFragment();
    } else {
      requestPermission();
    }

    AssetManager mAssets = this.getAssets();

    if (appList == null) {
      singletonAppList = SingletonAppList.getInstance();
      getAppList();
    }

    LOGGER.d(appListText);

    inputSize = getIntent().getIntExtra("InputSize",300);
    LOGGER.i("Input Size: "+ inputSize);

    MIN_MATCH_COUNT = 10*Math.round(30*inputSize/4032);

    fastDebug = getIntent().getBooleanExtra("FastDebug", false);
    overallTimes = new long[CAPTURE_TIMEOUT+1];
    detectionTimes = new long[CAPTURE_TIMEOUT+1];

    // creating an instance of the MrObjectManager
    if (manager == null) manager = new MrObjectManager();

    // checking NetworkMode and setting remote URL. This only works for the _withNetwork activity.
    try {
      NetworkMode = getIntent().getStringExtra("NetworkMode");
      if (NetworkMode!=null) {
        if (NetworkMode.equals("REMOTE_PROCESS")) {
        LOGGER.i("NetworkMode: " + NetworkMode);
        remoteUrl = getIntent().getStringExtra("RemoteURL");
        }
      }


      // network activity
      mNetworkFragment = NetworkFragment.getInstance(getSupportFragmentManager(), "http://"+ remoteUrl +":8081");
      mNetworkFragment.startServer(8081, mAssets);
      mNetworkFragment.setServerListener(this);
    } catch(Exception e) {
      e.printStackTrace();
    }


    //LOGGER.i("DataGathering, Image, Number of Apps, Frame Size, Overall Frame Processing (ms), Detection Time (ms), Number of hits");
    logWriter = singletonAppList.getWriter();

    utilityHit = 0; // initializing utilityHit
  }

  protected void getAppList(){
    LOGGER.d("Getting a new list from SingletonAppList");
    appList = singletonAppList.getList();
    appListText = singletonAppList.getListText();
  }

  private byte[] lastPreviewFrame;

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  protected int getLuminanceStride() {
    return yRowStride;
  }

  protected byte[] getLuminance() {
    return yuvBytes[0];
  }

  /**
   * Callback for android.hardware.Camera API
   */
  @Override
  public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    if (isProcessingFrame) {
      LOGGER.w("Dropping frame!");
      return;
    }

    try {
      // Initialize the storage bitmaps once when the resolution is known.
      if (rgbBytes == null) {
        Camera.Size previewSize = camera.getParameters().getPreviewSize();
        previewHeight = previewSize.height;
        previewWidth = previewSize.width;
        rgbBytes = new int[previewWidth * previewHeight];
        onPreviewSizeChosen(new Size(previewSize.width, previewSize.height), 90);
      }
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      return;
    }

    isProcessingFrame = true;
    lastPreviewFrame = bytes;
    yuvBytes[0] = bytes;
    yRowStride = previewWidth;

    imageConverter =
        new Runnable() {
          @Override
          public void run() {
            ImageUtils.convertYUV420SPToARGB8888(bytes, previewWidth, previewHeight, rgbBytes);
          }
        };

    postInferenceCallback =
        new Runnable() {
          @Override
          public void run() {
            camera.addCallbackBuffer(bytes);
            isProcessingFrame = false;
          }
        };

    if (appList != null) processImage();
  }

  /**
   * Callback for Camera2 API
   */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    //We need wait until we have some size from onPreviewSizeChosen
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();

      if (image == null) {
        return;
      }

      if (isProcessingFrame) {
        image.close();
        return;
      }
      isProcessingFrame = true;
      Trace.beginSection("imageAvailable");
      final Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);
      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();

      imageConverter =
          new Runnable() {
            @Override
            public void run() {
              ImageUtils.convertYUV420ToARGB8888(
                  yuvBytes[0],
                  yuvBytes[1],
                  yuvBytes[2],
                  previewWidth,
                  previewHeight,
                  yRowStride,
                  uvRowStride,
                  uvPixelStride,
                  rgbBytes);
            }
          };

      postInferenceCallback =
          new Runnable() {
            @Override
            public void run() {
              image.close();
              isProcessingFrame = false;
            }
          };

      if (appList != null) processImage();
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }

/*  private void startBackgroundThread() {
    handlerThread = new HandlerThread("inference");
    handlerThread.start();
    handler = new Handler(handlerThread.getLooper());
  }*/

/*  private void stopBackgroundThread() {
    handlerThread.quitSafely();
    try {
      handlerThread.join();
      handlerThread = null;
      handler = null;
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }*/

  @Override
  public synchronized void onStart() {
    LOGGER.d("onStart " + this);
    super.onStart();

    //startBackgroundThread();
    manager.generateList();
    utilityHit = 0;
  }

  @Override
  public synchronized void onResume() {
    LOGGER.d("onResume " + this);
    super.onResume();

    //startBackgroundThread();
    manager.generateList();
  }

  @Override
  public synchronized void onPause() {
    LOGGER.d("onPause " + this);

    if (!isFinishing()) {
      LOGGER.d("Requesting finish");
      finish();
    }

    //stopBackgroundThread();
    ProcessManager.cancelAll();


    super.onPause();
    appList = null;
    manager.storeList();
  }

  @Override
  public synchronized void onStop() {
    LOGGER.d("onStop " + this);

    ProcessManager.cancelAll();
    super.onStop();
  }

  @Override
  public synchronized void onDestroy() {
    LOGGER.d("onDestroy " + this);

    ProcessManager.cancelAll();
    super.onDestroy();
  }

/*  protected synchronized void runInThread() {

  }*/

  @Override
  public void onRequestPermissionsResult(
      final int requestCode, final String[] permissions, final int[] grantResults) {
    if (requestCode == PERMISSIONS_REQUEST) {
      if (grantResults.length > 0
          && grantResults[0] == PackageManager.PERMISSION_GRANTED
          && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
        setFragment();
      } else {
        requestPermission();
      }
    }
  }

  private boolean hasPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
          checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
    } else {
      return true;
    }
  }

  private void requestPermission() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
      if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
          shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
        Toast.makeText(MrThreadedCameraActivity.this,
            "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
      }
      requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
    }
  }

  // Returns true if the device supports the required hardware level, or better.
  private boolean isHardwareLevelSupported(
      CameraCharacteristics characteristics, int requiredLevel) {
    int deviceLevel = characteristics.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
    if (deviceLevel == CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_LEGACY) {
      return requiredLevel == deviceLevel;
    }
    // deviceLevel is not LEGACY, can use numerical sort
    return requiredLevel <= deviceLevel;
  }

  private String chooseCamera() {
    final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
    try {
      for (final String cameraId : manager.getCameraIdList()) {
        final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

        // We don't use a front facing camera in this sample.
        final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
        if (facing != null && facing == CameraCharacteristics.LENS_FACING_FRONT) {
          continue;
        }

        final StreamConfigurationMap map =
            characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

        if (map == null) {
          continue;
        }

        // Fallback to camera1 API for internal cameras that don't have full support.
        // This should help with legacy situations where using the camera2 API causes
        // distorted or otherwise broken previews.
        useCamera2API = (facing == CameraCharacteristics.LENS_FACING_EXTERNAL)
            || isHardwareLevelSupported(characteristics, 
                                        CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL_FULL);
        LOGGER.i("Camera API lv2?: %s", useCamera2API);
        return cameraId;
      }
    } catch (CameraAccessException e) {
      LOGGER.e(e, "Not allowed to access camera");
    }

    return null;
  }

  protected void setFragment() {
    String cameraId = chooseCamera();

    Fragment fragment;
    if (useCamera2API) {
      CameraConnectionFragment camera2Fragment =
          CameraConnectionFragment.newInstance(
              new CameraConnectionFragment.ConnectionCallback() {
                @Override
                public void onPreviewSizeChosen(final Size size, final int rotation) {
                  previewHeight = size.getHeight();
                  previewWidth = size.getWidth();
                  MrThreadedCameraActivity.this.onPreviewSizeChosen(size, rotation);
                }
              },
              this,
              getLayoutId(),
              getDesiredPreviewFrameSize());

      camera2Fragment.setCamera(cameraId);
      fragment = camera2Fragment;
    } else {
      fragment =
          new LegacyCameraConnectionFragment(this, getLayoutId(), getDesiredPreviewFrameSize());
    }

    getFragmentManager()
        .beginTransaction()
        .replace(R.id.container, fragment)
        .commit();
  }

  protected void fillBytes(final Plane[] planes, final byte[][] yuvBytes) {
    // Because of the variable row stride it's not possible to know in
    // advance the actual necessary dimensions of the yuv planes.
    for (int i = 0; i < planes.length; ++i) {
      final ByteBuffer buffer = planes[i].getBuffer();
      if (yuvBytes[i] == null) {
        LOGGER.d("Initializing buffer %d at size %d", i, buffer.capacity());
        yuvBytes[i] = new byte[buffer.capacity()];
      }
      buffer.get(yuvBytes[i]);
    }
  }

  public boolean isDebug() {
    return debug;
  }

  public void requestRender() {
    final OverlayView overlay = (OverlayView) findViewById(R.id.tracking_overlay);
    if (overlay != null) {
      overlay.postInvalidate();
    }
  }

  public void addCallback(final OverlayView.DrawCallback callback) {
    final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
    if (overlay != null) {
      overlay.addCallback(callback);
    }
  }

  public void onSetDebug(final boolean debug) {}

  @Override
  public boolean onKeyDown(final int keyCode, final KeyEvent event) {
    if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
      debug = !debug;
      requestRender();
      onSetDebug(debug);
      return true;
    }
    return super.onKeyDown(keyCode, event);
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  protected int getScreenOrientation() {
    switch (getWindowManager().getDefaultDisplay().getRotation()) {
      case Surface.ROTATION_270:
        return 270;
      case Surface.ROTATION_180:
        return 180;
      case Surface.ROTATION_90:
        return 90;
      default:
        return 0;
    }
  }

  @Override // part of the network listener
  public void setReceiveFlag(boolean value){
    receiveFlag = value;
  }

  @Override // part of the network listener
  public void uploadComplete(){
    // Do something here.
  }

  @Override // part of the network listener
  public void receivedFromNetwork(List<XmlOperator.XmlObject> objects){
    for (XmlOperator.XmlObject object: objects) {
      manager.processObject(object);
    }
  }

  protected abstract void processImage();
  protected abstract void onPreviewSizeChosen(final Size size, final int rotation);
  protected abstract int getLayoutId();
  protected abstract Size getDesiredPreviewFrameSize();
}
