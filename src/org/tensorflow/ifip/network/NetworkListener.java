package org.tensorflow.ifip.network;

import java.util.List;

/**
 * Created by deg032 on 22/2/18.
 */

public interface NetworkListener {

    void setReceiveFlag(boolean value);

    void uploadComplete();

    void receivedFromNetwork(List<XmlOperator.XmlObject> objects);
}
