package org.tensorflow.demo.orientation;

import android.graphics.RectF;

import org.tensorflow.demo.simulator.App;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by deg032 on 18/6/18.
 *
 * A singleton global list for identity preservation while tracking.
 */


public class GlobalTrackedObjectsList {

    private static final GlobalTrackedObjectsList instance = new GlobalTrackedObjectsList();

    // Private constructor prevents instantiation from other classes
    private GlobalTrackedObjectsList() {
    }

    private List<GlobalTrackedObject> list = new ArrayList<>();

    private class GlobalTrackedObject {
        RectF location;
        float detectionConfidence;
        int color;
        String title;
        int hits = 0;

        GlobalTrackedObject(RectF location, float detectionConfidence, int color, String title, int hits){
            this.location = location;
            this.detectionConfidence = detectionConfidence;
            this.color = color;
            this.title = title;
            this.hits = hits;
        }

        GlobalTrackedObject(GlobalTrackedObject object){
            this.location = object.location;
            this.detectionConfidence = object.detectionConfidence;
            this.color = object.color;
            this.title = object.title;
            this.hits = object.hits;
        }
    }

    public static GlobalTrackedObjectsList getInstance() {
        return instance;
    }

    public List<GlobalTrackedObject> getList() {
        return list;
    }

    public void add(RectF location, float detectionConfidence, int color, String title, int hits) {
        list.add(new GlobalTrackedObject(location, detectionConfidence, color, title, hits));
    }

    public void refreshList(){
        for (final GlobalTrackedObject object: list){

        }
    }
}
