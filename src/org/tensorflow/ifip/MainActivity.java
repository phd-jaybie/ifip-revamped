package org.tensorflow.ifip;

import android.app.Activity;
import android.app.AlertDialog;
import android.app.PendingIntent;
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
import android.support.v4.app.NotificationCompat;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.tensorflow.ifip.env.Logger;
import org.tensorflow.ifip.initializer.ObjectReferenceList;
import org.tensorflow.ifip.initializer.ReferenceObject;
import org.tensorflow.ifip.phd.MrDetectorActivity;
import org.tensorflow.ifip.phd.MrIfipDetectorActivity;
import org.tensorflow.ifip.phd.MrIfipDetectorActivityWithNetwork;
import org.tensorflow.ifip.phd.MrIfipNullActivity;
import org.tensorflow.ifip.phd.MrInitializeDemoDetectorActivity;
import org.tensorflow.ifip.phd.MrNullActivity;
import org.tensorflow.ifip.phd.MrThreadedDemoDetectorActivity;
import org.tensorflow.ifip.phd.ProtectedMrDetectorActivity;
import org.tensorflow.ifip.phd.ProtectedMrDetectorActivityWithNetwork;
import org.tensorflow.ifip.simulator.App;
import org.tensorflow.ifip.simulator.AppRandomizer;
import org.tensorflow.ifip.simulator.Randomizer;
import org.tensorflow.ifip.simulator.SingletonAppList;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by deg032 on 9/2/18.
 */

public class MainActivity extends Activity implements AdapterView.OnItemSelectedListener {
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

    private TextView textView;
    private EditText numberText;
    private EditText urlStringView;
    private EditText captureSizeView;
    private ListView listview;
    private Switch debugSwitch; // This switch just tells the processing activities if captures are limited or not.
    private Switch threadSwitch; // This switch just tells the processing activities if threaded or not.
    private Switch fixedAppsSwitch; // This switch just tells the app randomizer to create a fixed set of apps.
    private Switch networkSwitch; // This switch just tells whether the detection is local or remote.
    private Spinner modeSpinner;
    private Spinner remoteSpinner;

    /**
     * Variable @operatingMode default is SIFT.
     * To avoid null references when @modeSpinner is not set.
     */
    private static String operatingMode = "SIFT";

    private String NetworkMode = "REMOTE_PROCESS"; // default to remote when Network activity is clicked.
    private boolean FastDebug = false;
    private boolean Threading = false;
    private boolean FixedApps = false;
    private String remoteUrl = "";
    private int inputSize = 500;
    private String RemoteMode = "Co-located";

    public final String firstMessage = "Generate App list first.";
    public final String firstMessageDemo= "Initialize object references first.";

    private static TaskStackBuilder backStack;

    static {
/*        LOGGER.i("DataGatheringAverage, Image, Number of Apps, Frame Size, " +
                "Overall Frame Processing (ms), Detection Time (ms), " +
                "Number of hits, Secret hits (, Latent Privacy Hit) ");*/

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
        setContentView(R.layout.activity_main_ifip);

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
        // This snippet if for logging to file.

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

        textView = (TextView) findViewById(R.id.generate_textView);
        numberText = (EditText) findViewById(R.id.number_of_apps);
        captureSizeView = (EditText) findViewById(R.id.capture_size);
        urlStringView = (EditText) findViewById(R.id.remote_url);
        debugSwitch = (Switch) findViewById(R.id.debug_toggle);
        threadSwitch = (Switch) findViewById(R.id.thread_toggle);
        fixedAppsSwitch = (Switch) findViewById(R.id.fixed_apps_toggle);
        networkSwitch = (Switch) findViewById(R.id.network_toggle);
        singletonAppList = SingletonAppList.getInstance();
        objectReferenceList = ObjectReferenceList.getInstance();
        randomizer = AppRandomizer.create();

        modeSpinner = (Spinner) findViewById(R.id.mode_spinner);
        modeSpinner.setOnItemSelectedListener(this);

        remoteSpinner = (Spinner) findViewById(R.id.remote_spinner);
        remoteSpinner.setOnItemSelectedListener(this);

        ArrayAdapter<CharSequence> modeAdapter = ArrayAdapter.createFromResource(this,
                R.array.mode_array, android.R.layout.simple_spinner_item);
        modeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner.setAdapter(modeAdapter);

        ArrayAdapter<CharSequence> remoteAdapter = ArrayAdapter.createFromResource(this,
                R.array.remote_array, android.R.layout.simple_spinner_item);
        remoteAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        remoteSpinner.setAdapter(remoteAdapter);

        listview = (ListView) findViewById(R.id.listview);

        final ArrayList<ReferenceObject> list = new ArrayList<>(objectReferenceList.getList());

        if (!list.isEmpty()) {

            final StableArrayAdapter adapter =
                    new StableArrayAdapter(this, R.layout.list_object, list);
            listview.setAdapter(adapter);

            listview.setOnItemClickListener(new AdapterView.OnItemClickListener() {

                @Override
                public void onItemClick(AdapterView<?> parent, final View view,
                                        int position, long id) {

                    ReferenceObject object = list.get(position);
                    object.toggleSensitivity();
                    list.set(position,object);
                    adapter.notifyDataSetChanged();
                    view.setAlpha(1);
                }

            });
        }
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {

/*    if (pos == 0) {
        //do nothing, first item is a hint
        return;
    } else {*/
        switch (parent.getId()) {
            case R.id.mode_spinner:
                String sMode = parent.getItemAtPosition(pos).toString();
                switch (sMode) {
                    case "SIFT":
                        operatingMode = "SIFT";
                        break;
                    case "ORB":
                        operatingMode = "ORB";
                        break;
                    case "TF":
                        operatingMode = "TF";
                        break;
                    case "MARKER":
                        operatingMode = "MARKER";
                        break;
                    default:
                        operatingMode = "SIFT";
                        break;
                }
                // Showing selected spinner item
                //Toast.makeText(parent.getContext(), "Selected Mode: " + sMode, Toast.LENGTH_LONG).show();
                break;

            case R.id.remote_spinner:
                String rMode = parent.getItemAtPosition(pos).toString();
                switch (rMode) {
                    case "Co-located":
                        RemoteMode = "Co-located";
                        break;
                    case "Remote Edge":
                        RemoteMode = "Remote Edge";
                        break;
                    case "Cloud":
                        RemoteMode = "Cloud";
                        break;
                    default:
                        RemoteMode = "Co-located";
                        break;
                }
                // Showing selected spinner item
                //Toast.makeText(parent.getContext(), "Selected Remote: " + rMode, Toast.LENGTH_LONG).show();
                break;
        }

    //}

    }

    public void onNothingSelected(AdapterView<?> parent) {
        //MIN_MATCH_COUNT = 60;
        //ResolutionDivider = 2.4;
        operatingMode = "SIFT";
        RemoteMode = "Co-located";

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

            if (object.getSensitivity()) {
                String description =  object.getTitle() + ", sensitive";
                holder.text.setText(description);
                holder.text.setTextColor(Color.RED);
            } else {
                String description =  object.getTitle() + ", not sensitive";
                holder.text.setText(description);
                holder.text.setTextColor(Color.DKGRAY);
            }

            Matrix matrix = new Matrix();
            matrix.postRotate(90);

            Bitmap listImage = Bitmap.createScaledBitmap(object.getReferenceImage(),
                    100, 100, false);

            holder.image.setImageBitmap(Bitmap.createBitmap(listImage,
                    0,0,
                    listImage.getWidth(), listImage.getHeight(),
                    matrix, true)
            );

            return rowView;
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

        writeToTextView(message);

        if (FixedApps) {
            generateFixedApps(numberOfApps);
        } else {
            generateApps(numberOfApps);
        }

    }

    private void generateFixedApps(final int numberOfApps) {
        runInBackground(new Runnable() {
            @Override
            public void run() {
                writeToTextView("Generating a fixed app list.");

                appList = randomizer.fixedAppGenerator(getApplicationContext(), numberOfApps);

                String appLogMessage = "App list:\n";
                for (App app : appList) {
                    appLogMessage = appLogMessage + app.getName() + "\n";
                }
                LOGGER.i(appLogMessage);
                appListText = appLogMessage;

                //writeToTextView(appLogMessage);
                singletonAppList.setList(appList);
                singletonAppList.setListText(appListText);
            }
        });
    }

    private void generateApps(final int numberOfApps) {
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

                writeToTextView(appLogMessage);
                singletonAppList.setList(appList);
                singletonAppList.setListText(appListText);
            }
        });
    }

    private boolean checkList(){

        String captureSizeViewText = captureSizeView.getText().toString();
        if (captureSizeViewText.isEmpty()) inputSize = 300;
        else inputSize = Integer.valueOf(captureSizeViewText);

        if (singletonAppList.getList().isEmpty()) {
            writeToTextView(firstMessage);
            return false;
        /*} else if (networkSwitch.isChecked() && (remoteUrl == null)) {// || !URLUtil.isValidUrl(remoteUrl)) ) {
            writeToTextView("No or Invalid URL for remote.");
            return false;*/
        } else {
            writeToTextView(singletonAppList.getListText());
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
            writeToTextView(firstMessageDemo);
            return false;
        }

        return true;

    }

    private void writeToTextView(final String message){

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (textView == null) textView = (TextView) findViewById(R.id.generate_textView);
                textView.setText(message);
            }
        });
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

        if (!checkList()) generateFixedApps(1); //return; is default

        Intent detectorIntent = new Intent(this, MrIfipNullActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);
        detectorIntent.putExtra("OperatingMode", operatingMode);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();

    }

    public void mrDetectionIntent(View view){

        if (!checkList()) return;

        if (Threading) {
            Intent detectorIntent = new Intent(this, MrThreadedDemoDetectorActivity.class);
            detectorIntent.putExtra("InputSize", inputSize);
            detectorIntent.putExtra("FastDebug", FastDebug);
            startActivity(detectorIntent);
        } else {
            Intent detectorIntent = new Intent(this, MrDetectorActivity.class);
            detectorIntent.putExtra("InputSize", inputSize);
            detectorIntent.putExtra("FastDebug", FastDebug);
            startActivity(detectorIntent);
        }

    }

    public void mrDetectionIntentProtected(View view){

        if (!checkList()) return;

        Intent detectorIntent = new Intent(this, ProtectedMrDetectorActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);
        startActivity(detectorIntent);

    }

    public void mrDetectionIntentWithSharing(View view){

        if (!checkList()) generateFixedApps(1); //return; is default

        Intent detectorIntent = new Intent(this, MrIfipDetectorActivityWithNetwork.class);
        detectorIntent.putExtra("NetworkMode",NetworkMode);
        detectorIntent.putExtra("OperatingMode", operatingMode);
        detectorIntent.putExtra("RemoteURL",remoteUrl);
        detectorIntent.putExtra("RemoteMode", RemoteMode);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();
        //startActivity(detectorIntent);

    }

    public void mrDemoDetectionIntent(View view){

        if (!checkListDemo()) return;

        Intent detectorIntent = new Intent(this, MrDetectorActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("FastDebug", FastDebug);

        backStack.addNextIntentWithParentStack(detectorIntent);
        backStack.startActivities();

        //startActivity(detectorIntent);

    }

    public void mrIfipDetectionIntent(View view){

        //if (!checkListDemo()) return;
        // Instead of using the checkList module, we generate our own list instead with size 1.
/*        generateFixedApps(1);

        String captureSizeViewText = captureSizeView.getText().toString();
        if (captureSizeViewText.isEmpty()) inputSize = 300;
        else inputSize = Integer.valueOf(captureSizeViewText);*/

        if (!checkList()) generateFixedApps(1); //return; is default

        Intent detectorIntent = new Intent(this, MrIfipDetectorActivity.class);
        detectorIntent.putExtra("InputSize", inputSize);
        detectorIntent.putExtra("OperatingMode", operatingMode);
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

        initialize();
        startBackgroundThread();
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        initialize();
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
