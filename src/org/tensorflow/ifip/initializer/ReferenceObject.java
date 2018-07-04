package org.tensorflow.ifip.initializer;

import android.graphics.Bitmap;

/**
 * Created by deg032 on 21/6/18.
 */

public class ReferenceObject {
    private Bitmap referenceImage;
    private String title;
    private boolean sensitive = false;

    ReferenceObject(Bitmap referenceImage, String title){
        this.referenceImage = referenceImage;
        this.title = title;
    }

    ReferenceObject(ReferenceObject object){
        this.referenceImage = object.referenceImage;
        this.title = object.title;
    }

    public Bitmap getReferenceImage(){
        return this.referenceImage;
    }

    public String getTitle(){
        return this.title;
    }

    public void toggleSensitivity() {
        this.sensitive = !this.sensitive;
    }

    public boolean getSensitivity(){
        return this.sensitive;
    }

}
