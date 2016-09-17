package edu.brown.cs.burlap.preprocess;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * A PreProcessor for the Atari games using Deep-Mind's specifications.
 *
 * @author Melrose Roderick.
 */
public class ALEPreProcessor implements PreProcessor {

    static final int scaleWidth = 84;
    static final int scaleHeight = 84;

    public ALEPreProcessor() {}

    @Override
    public void convertScreenToData(Mat screen, BytePointer data) {

        Mat gray = new Mat(screen.rows(), screen.cols(), CV_8UC1);
        cvtColor(screen, gray, COLOR_BGR2GRAY);

        Mat downsample = new Mat(scaleHeight, scaleWidth, CV_8UC1, data);
        resize(gray, downsample, new Size(scaleWidth, scaleHeight));
    }

    @Override
    public void convertDataToInput(BytePointer data, FloatPointer input, long size) {

        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Size is too large to create an opencv Mat");
        }

        int dataSize = outputSize() * (int)size;

        Mat mat = new Mat(1, dataSize, CV_8U, data);
        Mat floatMat = new Mat(1, dataSize, CV_32F, (new BytePointer(input)).position(input.position() * input.sizeof()));

        mat.convertTo(floatMat, CV_32F, 1/255.0, 0);
    }

    @Override
    public int outputSize() {
        return 84*84;
    }
}
