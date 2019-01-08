package org.tensorflow.demo;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.TaskStackBuilder;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.tensorflow.demo.arcore.MrDemoDetectorWithARCoreActivity;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.initializer.ObjectReferenceList;
import org.tensorflow.demo.initializer.ReferenceObject;
import org.tensorflow.demo.phd.MrDemoDetectorActivity;
import org.tensorflow.demo.phd.MrDetectorActivity;
import org.tensorflow.demo.phd.MrInitializeDemoDetectorActivity;
import org.tensorflow.demo.phd.MrNullActivity;
import org.tensorflow.demo.phd.threaded.MrThreadedDemoDetectorActivity;
import org.tensorflow.demo.phd.abstraction.ProtectedMrDetectorActivity;
import org.tensorflow.demo.phd.abstraction.ProtectedMrDetectorActivityWithNetwork;
import org.tensorflow.demo.simulator.App;
import org.tensorflow.demo.simulator.AppRandomizer;
import org.tensorflow.demo.simulator.Randomizer;
import org.tensorflow.demo.simulator.SingletonAppList;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by deg032 on 9/2/18.
 */

public class MainActivity extends Activity {
    private static final Logger LOGGER = new Logger();
    private static String logFile;
    private static PrintWriter writer;

    private Handler handler;
    private HandlerThread handlerThread;

    private Randomizer randomizer;

    private int numberOfApps;
    public List<App> appList = new ArrayList<>();
    public String appListText;

    private SingletonAppList singletonAppList;

    private ObjectReferenceList objectReferenceList;

    private EditText numberText;
    private EditText urlStringView;
    private EditText captureSizeView;
    private ListView listview;
    private Switch debugSwitch; // This switch just tells the processing activities if captures are limited or not.
    private Switch threadSwitch; // This switch just tells the processing activities if threaded or not.
    private Switch fixedAppsSwitch; // This switch just tells the app randomizer to create a fixed set of apps.
    private Switch networkSwitch; // This switch just tells whether the detection is local or remote.

    private String NetworkMode = "LOCAL";
    private boolean FastDebug = false;
    private boolean Threading = false;
    private boolean FixedApps = false;
    private String remoteUrl = null;
    private int inputSize = 500;

    public final String firstMessage = "Generate App list first.";
    public final String firstMessageDemo= "Initialize object references first.";

    private static TaskStackBuilder backStack;

    static {
        LOGGER.i("DataGatheringAverage, Image, Number of Apps, Frame Size, " +
                "Overall Frame Processing (ms), Detection Time (ms), " +
                "Number of hits, Secret hits (, Latent Privacy Hit) ");

        if(!OpenCVLoader.initDebug()){
            LOGGER.d("OpenCV not loaded");
        } else {
            LOGGER.d("OpenCV loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState){
        LOGGER.d("onCreate " + this);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_demo);

        initialize(); //Initializes the views

/*        if (isNetworkConnected()) {
            ProgressDialog mProgressDialog = new ProgressDialog(this);
            mProgressDialog.setMessage("Please wait...");
            mProgressDialog.setCancelable(true);
            mProgressDialog.show();
        } else {
            noConnection();
        }*/
        backStack = TaskStackBuilder.create(this);

    }

/*    private boolean isNetworkConnected() {
        ConnectivityManager connMgr = (ConnectivityManager)
                getSystemService(Context.CONNECTIVITY_SERVICE); // 1
        NetworkInfo networkInfo = connMgr.getActiveNetworkInfo(); // 2
        return networkInfo != null && networkInfo.isConnected(); // 3
    }

    private void noConnection(){
        new AlertDialog.Builder(this)
                .setTitle("No Internet Connection")
                .setMessage("It looks like your internet connection is off. Please turn it " +
                        "on and try again")
                .setPositiveButton(android.R.string.ok, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                    }
                });//.setIcon(android.R.drawable.ic_dialog_alert).show();
    }*/

    private void initialize() {

/*
        Timestamp timestamp = new Timestamp(System.currentTimeMillis());

        logFile = "log-" + timestamp.toString() + ".txt";

        try {
            Context context = this.getApplicationContext();
            File file = new File(context.getFilesDir(), logFile);
            FileWriter writer = new FileWriter(file);
            writer.write("DataGathering, Image, Number of Apps, Frame Size," +
                    "Overall Frame Processing (ms), Detection Time (ms)");
            singletonAppList.setWriter(writer);
        } catch (Exception e) {
            e.printStackTrace();
        }
*/

        numberText = (EditText) findViewById(R.id.number_of_apps);
        captureSizeView = (EditText) findViewById(R.id.capture_size);
        urlStringView = (EditText) findViewById(R.id.remote_url);
        debugSwitch = (Switch) findViewById(R.id.debug_toggle);
        threadSwitch = (Switch) findViewById(R.id.thread_toggle);
        fixedAppsSwitch = (Switch) findViewById(R.id.fixed_apps_toggle);
        networkSwitch = (Switch) findViewById(R.id.network_toggle);
        singletonAppList = SingletonAppList.getInstance();
        objectReferenceList = ObjectReferenceList.getInstance();

        listview = (ListView) findViewById(R.id.listview);

        addVirtualObjects();

        final ArrayList<ReferenceObject> list = new ArrayList<>(objectReferenceList.getList());

        if (!list.isEmpty()) {

            final StableArrayAdapter adapter =
                    new StableArrayAdapter(this, R.layout.list_object, list);
            listview.setAdapter(adapter);

            listview.setOnItemClickListener((parent, view, position, id) -> {

                ReferenceObject object = list.get(position);
                object.toggleSensitivity();
                list.set(position,object);
                adapter.notifyDataSetChanged();
                view.setAlpha(1);
            });

/*            listview.setOnItemClickListener(new AdapterView.OnItemClickListener() {

                @Override
                public void onItemClick(AdapterView<?> parent, final View view,
                                        int position, long id) {

                    ReferenceObject object = list.get(position);
                    object.toggleSensitivity();
                    list.set(position,object);
                    adapter.notifyDataSetChanged();
                    view.setAlpha(1);
                }

            });*/
        }
    }

    private class StableArrayAdapter extends ArrayAdapter<ReferenceObject> {

        private final Context context;
        private final List<ReferenceObject> list;

        class ViewHolder {
            public TextView text;
            public ImageView image;
        }

        public StableArrayAdapter(Context context, int textViewResourceId,
                                  List<ReferenceObject> objects) {
            super(context, textViewResourceId, objects);
            this.context = context;
            this.list = objects;
        }

        @Override
        public View getView(int position, View convertView, ViewGroup parent) {
            View rowView = convertView;
            // reuse views
            if (rowView == null) {
                rowView = getLayoutInflater().inflate(R.layout.list_object, null);
                // configure view holder
                ViewHolder viewHolder = new ViewHolder();
                viewHolder.text = (TextView) rowView.findViewById(R.id.title);
                viewHolder.image = (ImageView) rowView
                        .findViewById(R.id.icon);
                rowView.setTag(viewHolder);
            }

            // fill data
            ViewHolder holder = (ViewHolder) rowView.getTag();
            ReferenceObject object = list.get(position);

            if (object.isSensitive()) {
                String description =  object.getTitle() + ", sensitive";
                holder.text.setText(description);
                holder.text.setTextColor(Color.RED);
            } else {
                String description =  object.getTitle() + ", not sensitive";
                holder.text.setText(description);
                holder.text.setTextColor(Color.DKGRAY);
            }

            if (object.isVirtual()) {
                holder.image.setImageResource(object.getVirtualId());
            } else {

                Matrix matrix = new Matrix();
                matrix.postRotate(90);

                Bitmap listImage = Bitmap.createScaledBitmap(object.getReferenceImage(),
                        100, 100, false);
                holder.image.setImageBitmap(Bitmap.createBitmap(listImage,
                        0,0,
                        listImage.getWidth(), listImage.getHeight(),
                        matrix, true)
                );
            }

            return rowView;
        }

    }

    private void addVirtualObjects(){
        if (!objectReferenceList.isWithVirtualObjects()) {
            objectReferenceList.add(R.drawable.igloo_thumb, R.raw.igloo, "igloo");
            objectReferenceList.add(R.drawable.droid_thumb, R.raw.andy,"droid");
            objectReferenceList.add(R.drawable.house_thumb, R.raw.house,"house");
            objectReferenceList.setWithVirtualObjects(true);
        }
    }

    public void generateAppList(View view){

        String sNumberOfApps = numberText.getText().toString();

        if (sNumberOfApps.isEmpty()) return;

        numberOfApps = Integer.valueOf(sNumberOfApps);

        String message;

        if (debugSwitch.isChecked()) message = "Will only take a limited number captures.\n";
        else message = "Will capture continuously.\n";

        message = message + "Creating a " + numberOfApps + "-app list.\n";
        LOGGER.i(message);

        writeToToast(message);

        randomizer = AppRandomizer.create();

        if (FixedApps) {

            runInBackground(new Runnable() {
                @Override
                public void run() {
                    appList = randomizer.fixedAppGenerator(getApplicationContext(), numberOfApps);

                    String appLogMessage = "App list:\n";
                    for (App app : appList) {
                        appLogMessage = appLogMessage + app.getName() + "\n";
                    }
                    LOGGER.i(appLogMessage);
                    appListText = appLogMessage;

                    writeToToast(appLogMessage);
                    singletonAppList.setList(appList);
                    singletonAppList.setListText(appListText);
                }
            });

        } else {
            runInBackground(new Runnable() {
                @Override
                public void run() {
                    appList = randomizer.appGenerator(getApplicationContext(), numberOfApps);

                    String appLogMessage = "App list:\n";
                    for (App app : appList) {
                        appLogMessage = appLogMessage + app.getName() + "\n";
                    }
                    LOGGER.i(appLogMessage);
                    appListText = appLogMessage;

                    writeToToast(appLogMessage);
                    singletonAppList.setList(appList);
                    singletonAppList.setListText(appListText);
                }
            });
        }

    }

    private boolean checkList(){

        String captureSizeViewText = captureSizeView.getText().toString();
        if (captureSizeViewText.isEmpty()) inputSize = 300;
        else inputSize = Integer.valueOf(captureSizeViewText);

        if (singletonAppList.getList().isEmpty()) {
            writeToToast(firstMessage);
            return false;
        } else if (networkSwitch.isChecked() && (remoteUrl == null)) {// || !URLUtil.isValidUrl(remoteUrl)) ) {
            writeToToast("No or Invalid URL for remote.");
            return false;
        } else {
            writeToToast(singletonAppList.getListText());
            return true;
        }

    }

    private boolean checkListDemo(){

/*
        String captureSizeViewText = captureSizeView.getText().toString();
        if (captureSizeViewText.isEmpty()) inputSize = 300;
        else inputSize = Integer.valueOf(captureSizeViewText);
*/

        if (objectReferenceList.getList().isEmpty()) {
            writeToToast(firstMessageDemo);
            return false;
        }

        return true;

    }

    private void writeToToast(final String message){
        Toast toast = Toast.makeText(MainActivity.this, message,
                Toast.LENGTH_LONG);
        toast.show();
/*        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (textView == null) textView = (TextView) findViewById(R.id.generate_textView);
                textView.setText(message);
            }
        });*/
    }

    public void onFastDebug(View view){
        if (debugSwitch.isChecked()) {
            debugSwitch.setTextColor(Color.BLACK);
            FastDebug = true;
        } else {
            debugSwitch.setTextColor(Color.LTGRAY);
            FastDebug = false;
        }

    }

    public void onThread(View view){
        if (threadSwitch.isChecked()) {
            threadSwitch.setTextColor(Color.BLACK);
            Threading = true;
        } else {
            threadSwitch.setTextColor(Color.LTGRAY);
            Threading = false;
        }

    }

    public void onFixedApps(View view){
        if (fixedAppsSwitch.isChecked()) {
            fixedAppsSwitch.setTextColor(Color.BLACK);
            FixedApps = true;
        } else {
            fixedAppsSwitch.setTextColor(Color.LTGRAY);
            FixedApps = false;
        }

    }

    public void onNetworkProcess(View view){

        remoteUrl = urlStringView.getText().toString();

        if (networkSwitch.isChecked()) {
            LOGGER.i("Remote image processing.");
            networkSwitch.setTextColor(Color.BLACK);
            NetworkMode = "REMOTE_PROCESS";
        } else {
            LOGGER.i("Local image processing.");
            NetworkMode = "LOCAL";
            networkSwitch.setTextColor(Color.LTGRAY);
        }

    }

    public void mrNullIntent(View view){

        //if (!checkList()) return;

        Intent detectorIntent = new Intent(this, MrNullActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    public void mrDetectionIntent(View view){

        if (!checkList()) return;

        if (Threading) {
            Intent detectorIntent = new Intent(this, MrThreadedDemoDetectorActivity.class);
            detectorIntent.putExtra("InputSize", inputSize);
            detectorIntent.putExtra("FastDebug", FastDebug);

            //startActivity(detectorIntent);
            backStack.addNextIntentWithParentStack(detectorIntent);

        } else {
            Intent detectorIntent = new Intent(this, MrDetectorActivity.class);
            detectorIntent.putExtra("InputSize", inputSize);
            detectorIntent.putExtra("FastDebug", FastDebug);

            //startActivity(detectorIntent);
            backStack.addNextIntentWithParentStack(detectorIntent);

        }

        backStack.startActivities();

    }

    public void mrDetectionIntentProtected(View view){

        if (!checkList()) return;

        Intent detectorIntent = new Intent(this, ProtectedMrDetectorActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    public void mrDetectionIntentWithSharing(View view){

        if (!checkList()) return;

        Intent detectorIntent = new Intent(this, ProtectedMrDetectorActivityWithNetwork.class);
        detectorIntent.putExtra("NetworkMode",NetworkMode);
        detectorIntent.putExtra("RemoteURL",remoteUrl);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    public void mrDemoDetectionIntent(View view){

        if (!checkListDemo()) {
            return;
        }

        Intent detectorIntent = new Intent(this, MrDemoDetectorActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    public void mrDemoDetectionWithARCoreIntent(View view){

/*        if (!checkListDemo()) {
            return;
        }*/

        Intent detectorIntent = new Intent(this, MrDemoDetectorWithARCoreActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    public void mrDemoInitializeIntent(View view){

        Intent detectorIntent = new Intent(this, MrInitializeDemoDetectorActivity.class);
        //detectorIntent.putExtra("InputSize", inputSize);
        //detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    private void startBackgroundThread() {
        handlerThread = new HandlerThread("main activity");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }

    private void stopBackgroundThread() {
        handlerThread.quitSafely();
        try {
            handlerThread.join();
            handlerThread = null;
            handler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();

        //();
        startBackgroundThread();
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        //initialize();
        startBackgroundThread();
    }

    @Override
    public synchronized void onPause() {
        LOGGER.d("onPause " + this);

        if (!isFinishing()) {
            LOGGER.d("Requesting finish");
            finish();
        }

        stopBackgroundThread();

        super.onPause();
        if (writer!=null) writer.close();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

}
