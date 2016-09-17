package edu.brown.cs.burlap.preprocess;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * An interface for converting an OpenCV matrix to a savable byte array,
 * then to a float array for running through Caffe.
 *
 * @author Melrose Roderick.
 */
public interface PreProcessor {
    void convertScreenToData(Mat screen, BytePointer data);
    void convertDataToInput(BytePointer data, FloatPointer input, long size);
    int outputSize();
}
