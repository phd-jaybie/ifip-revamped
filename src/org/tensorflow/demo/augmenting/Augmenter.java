package org.tensorflow.demo.augmenting;

import android.graphics.Canvas;

import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.phd.abstraction.MrObjectManager;

/**
 * Created by deg032 on 1/2/18.
 */

public class Augmenter extends MrObjectManager{

    private final Logger logger = new Logger();

    //private List<Pair<MrObject, Long>> liveMrObjects = new ArrayList<>();

    public synchronized void drawAugmentations (Canvas canvas){
        /**
         * Insert here actual drawing of augmentations of live, detected and tracked bject/s.
         */

        for (final MrObject mrObject: MrObjects) {

        }
    }

    public synchronized void trackResults(final byte[] frame, final long timestamp) {
        /**
         * Insert code here that handles the augmentation of detected/tracked objects.
         */
    }
}
