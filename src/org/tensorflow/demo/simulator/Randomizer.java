package org.tensorflow.demo.simulator;

import android.content.Context;

import java.util.List;

/**
 * Created by deg032 on 2/2/18.
 */

public interface Randomizer {

    List<App> appGenerator(Context context, int numberOfApps);

    List<App> fixedAppGenerator(Context context, int numberOfApps);

    //List<User> userGenerator(int numberOfUsers);

}
