package org.tensorflow.demo.initializer;

import android.graphics.Bitmap;

import com.google.ar.core.Anchor;

/**
 * Created by deg032 on 21/6/18.
 */

public class ReferenceObject {
    private Bitmap referenceImage;
    private String title;
    private boolean sensitive = false;
    private boolean virtual = false;
    private Integer virtualId;
    private Integer virtualRawResourceId;
    private String virtualAnchorId;
    private boolean virtualRendered = false;

    ReferenceObject(Bitmap referenceImage, String title){
        this.referenceImage = referenceImage;
        this.title = title;
    }

    ReferenceObject(ReferenceObject object){
        this.referenceImage = object.referenceImage;
        this.title = object.title;
    }

    /**
     * For adding a virtual object on to the reference list with toggable
     * @param virtualId
     * @param title
     */
    ReferenceObject(Integer virtualId,Integer virtualRawResourceId, String title){
        this.virtualId = virtualId;
        this.virtualRawResourceId = virtualRawResourceId;
        this.title = title;
        this.virtual = true;
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

    public boolean isSensitive(){
        return this.sensitive;
    }

    public boolean isVirtual(){
        return this.virtual;
    }

    public Integer getVirtualId() {
        return virtualId;
    }

    public String getVirtualAnchorId() {
        return virtualAnchorId;
    }

    public void setVirtualAnchorId(String virtualAnchorId) {
        this.virtualAnchorId = virtualAnchorId;
    }

    public Integer getVirtualRawResourceId() {
        return virtualRawResourceId;
    }

    public void setVirtualRendered(boolean virtualRendered) {
        this.virtualRendered = virtualRendered;
    }

    public boolean isVirtualRendered() {
        return virtualRendered;
    }
}
