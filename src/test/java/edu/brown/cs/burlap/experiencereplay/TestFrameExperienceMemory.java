package edu.brown.cs.burlap.experiencereplay;

import burlap.mdp.core.action.Action;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.burlap.ALEState;
import edu.brown.cs.burlap.action.ActionSet;
import edu.brown.cs.burlap.preprocess.PreProcessor;
import edu.brown.cs.burlap.vfa.StateVectorizor;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_8U;

/**
 * Created by maroderi on 7/19/16.
 */
public class TestFrameExperienceMemory {
    FloatPointer input;


    @Before
    public void setup() {
        Loader.load(opencv_core.class);
    }

    @After
    public void teardown() {

    }

    @Test
    public void TestSmall() {
        BytePointer data0 = new BytePointer((byte)0, (byte)0);
        BytePointer data1 = new BytePointer((byte)0, (byte)1);
        BytePointer data2 = new BytePointer((byte)2, (byte)3);
        BytePointer data3 = new BytePointer((byte)4, (byte)5);
        BytePointer data4 = new BytePointer((byte)6, (byte)7);
        BytePointer data5 = new BytePointer((byte)8, (byte)9);
        BytePointer data6 = new BytePointer((byte)10, (byte)11);
        BytePointer data7 = new BytePointer((byte)12, (byte)13);

        opencv_core.Mat frame1 = new opencv_core.Mat(1, 2, CV_8U, data1);
        opencv_core.Mat frame2 = new opencv_core.Mat(1, 2, CV_8U, data2);
        opencv_core.Mat frame3 = new opencv_core.Mat(1, 2, CV_8U, data3);
        opencv_core.Mat frame4 = new opencv_core.Mat(1, 2, CV_8U, data4);
        opencv_core.Mat frame5 = new opencv_core.Mat(1, 2, CV_8U, data5);
        opencv_core.Mat frame6 = new opencv_core.Mat(1, 2, CV_8U, data6);
        opencv_core.Mat frame7 = new opencv_core.Mat(1, 2, CV_8U, data7);

        ALEState aleState1 = new ALEState(frame1);
        ALEState aleState2 = new ALEState(frame2);
        ALEState aleState3 = new ALEState(frame3);
        ALEState aleState4 = new ALEState(frame4);
        ALEState aleState5 = new ALEState(frame5);
        ALEState aleState6 = new ALEState(frame6);
        ALEState aleState7 = new ALEState(frame7);


        input = new FloatPointer(2 * 2);

        ActionSet actionSet = new ActionSet(new String[]{"Action0"});
        Action action0 = actionSet.get(0);

        FrameExperienceMemory experienceMemory = new FrameExperienceMemory(5, 2, new TestPreprocessor(2), actionSet);
        FrameHistory state0 = experienceMemory.currentFrameHistory;
        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState1, 0, false));
        FrameHistory state1 = experienceMemory.currentFrameHistory;
        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState2, 0, false));
        FrameHistory state2 = experienceMemory.currentFrameHistory;
        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState3, 0, false));
        FrameHistory state3 = experienceMemory.currentFrameHistory;
        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState4, 0, false));
        FrameHistory state4 = experienceMemory.currentFrameHistory;

        compare(state0, experienceMemory, new BytePointer[]{data0, data0}, 2);
        compare(state1, experienceMemory, new BytePointer[]{data0, data1}, 2);
        compare(state2, experienceMemory, new BytePointer[]{data1, data2}, 2);
        compare(state3, experienceMemory, new BytePointer[]{data2, data3}, 2);
        compare(state4, experienceMemory, new BytePointer[]{data3, data4}, 2);

        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState5, 0, false));
        FrameHistory state5 = experienceMemory.currentFrameHistory;
        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState6, 0, false));
        FrameHistory state6 = experienceMemory.currentFrameHistory;
        experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState7, 0, false));
        FrameHistory state7 = experienceMemory.currentFrameHistory;

        compare(state3, experienceMemory, new BytePointer[]{data2, data3}, 2);
        compare(state4, experienceMemory, new BytePointer[]{data3, data4}, 2);
        compare(state5, experienceMemory, new BytePointer[]{data4, data5}, 2);
        compare(state6, experienceMemory, new BytePointer[]{data5, data6}, 2);
        compare(state7, experienceMemory, new BytePointer[]{data6, data7}, 2);
    }

    @Test
    public void TestRandom() {
        int replaySize = 50;
        int history = 4;
        int frameSize = 10;
        input = new FloatPointer(frameSize * history);

        ActionSet actionSet = new ActionSet(new String[]{"Action0"});
        Action action0 = actionSet.get(0);

        FrameExperienceMemory experienceMemory = new FrameExperienceMemory(replaySize, history, new TestPreprocessor(frameSize), actionSet);
        FrameHistory initialState = experienceMemory.currentFrameHistory;
        BytePointer data0 = new BytePointer(history);
        for (int f = 0; f < frameSize; f++) {
            data0.position(f).put((byte)0);
        }
        data0.position(0);
        List<BytePointer> dataList = new ArrayList<>();
        for (int h = 0; h < history; h++) {
            dataList.add(data0);
        }
        compare(initialState, experienceMemory, dataList.toArray(new BytePointer[history]), frameSize);

        List<List<BytePointer>> dataListList = new ArrayList<>();

        List<FrameHistory> states = new ArrayList<>();

        for (int n = 0; n < 100; n++) {
            for (int i = 0; i < replaySize; i++) {
                BytePointer data = new BytePointer(frameSize);
                for (int f = 0; f < frameSize; f++) {
                    byte d = (byte) (Math.random()*126.0);
                    data.position(f).put(d);
                }
                data.position(0);
                dataList.remove(0);
                dataList.add(data);

                opencv_core.Mat frame = new opencv_core.Mat(1, frameSize, CV_8U, data);
                ALEState aleState = new ALEState(frame);
                experienceMemory.addExperience(new EnvironmentOutcome(null, action0, aleState, 0, false));
                FrameHistory state = experienceMemory.currentFrameHistory;

                compare(state, experienceMemory, dataList.toArray(new BytePointer[history]), frameSize);

                if (i < dataListList.size()) {
                    dataListList.set(i, new ArrayList<>(dataList));
                    states.set(i, state);
                } else {
                    dataListList.add(new ArrayList<>(dataList));
                    states.add(state);
                }

                for (int k = 0; k < states.size(); k++) {
                    compare(states.get(k), experienceMemory, dataListList.get(k).toArray(new BytePointer[history]), frameSize);
                }
            }
        }
    }

    public void compare(FrameHistory state, StateVectorizor vectorizor, BytePointer[]dataArray, long outputSize) {
        vectorizor.vectorizeState(state, input);

        int i = 0;
        for (BytePointer data : dataArray) {
            for (int k = 0; k < outputSize; k++) {
                Assert.assertEquals(input.get(i), data.get(k), 1e-6);
                i++;
            }
        }
    }


    public class TestPreprocessor implements PreProcessor {
        int frameSize;

        public TestPreprocessor(int frameSize) {
            this.frameSize = frameSize;
        }

        @Override
        public void convertScreenToData(opencv_core.Mat screen, BytePointer data) {
            if (screen.data().address() == data.address()) {
                return;
            }

            BytePointer screenData = screen.data();
            data.limit(data.position() + frameSize).put(screen.data().limit(frameSize));
        }

        @Override
        public void convertDataToInput(BytePointer data, FloatPointer input, long size) {
            int dataSize = outputSize() * (int)size;

            opencv_core.Mat mat = new opencv_core.Mat(1, dataSize, CV_8U, data);
            opencv_core.Mat floatMat = new opencv_core.Mat(1, dataSize, CV_32F, (new BytePointer(input)).position(input.position() * input.sizeof()));

            mat.convertTo(floatMat, CV_32F, 1, 0);
        }

        @Override
        public int outputSize() {
            return frameSize;
        }
    }
}
