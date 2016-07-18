package edu.brown.cs.burlap.preprocess;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Created by MelRod on 5/27/16.
 */
public interface PreProcessor {
    void convertScreenToData(Mat screen, BytePointer data);
    void convertDataToInput(BytePointer data, FloatPointer input, long size);
    int outputSize();
}
