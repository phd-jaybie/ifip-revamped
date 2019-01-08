package org.tensorflow.demo.simulator;

import android.util.Pair;

import org.tensorflow.demo.Classifier;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by deg032 on 7/2/18.
 */

public class App implements Serializable {

    private final int id;
    private final String name;
    private Pair<String, String> method;
    private final String[] objectsOfInterest;
    private AppRandomizer.ReferenceImage reference;

    private final List<App.AppCallback> callbacks = new LinkedList<>();

    public App(int id, String name, Pair<String, String> method, String[] objects, AppRandomizer.ReferenceImage reference) {
        this.id = id;
        this.name = name;
        this.method = method;
        this.objectsOfInterest = objects;
        this.reference = reference;
    }

    public int getId() {
        return id;
    }

    public Pair<String, String> getMethod() {
        return method;
    }

    public String getName() {
        return name;
    }

    public String[] getObjectsOfInterest() {
        return objectsOfInterest;
    }

    public AppRandomizer.ReferenceImage getReference() {
        return reference;
    }

    public String toString(){
        String appString = Integer.toString(this.id);
        appString = appString + this.name;
        appString = appString + this.method;
        return appString;
    }

    public interface AppCallback {
        public void appCallback();
    }

    public void addCallback(final AppCallback callback) {
        callbacks.add(callback);
    }

    public synchronized void processAllCallbacks() {
        for (final AppCallback callback : callbacks) {
            callback.appCallback();
        }
    }

    public void process(Classifier.Recognition recognition, Long timestamp){
        // This is a sample method that an app's appCallback calls.
        // Each app can define their own appCallback methods.
    }

}
