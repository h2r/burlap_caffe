package edu.brown.cs.burlap.experiencereplay;

import burlap.debugtools.RandomFactory;
import burlap.mdp.auxiliary.StateMapping;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.burlap.ALEState;
import edu.brown.cs.burlap.action.ActionSet;
import edu.brown.cs.burlap.preprocess.PreProcessor;
import edu.brown.cs.burlap.vfa.StateVectorizor;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.bytedeco.javacpp.opencv_core.Mat;

/**
 * Created by maroderi on 6/20/16.
 */
public class FrameExperienceMemory implements SavableExperienceMemory, StateVectorizor, StateMapping, Serializable {

    public transient BytePointer frameMemory;
    public transient PreProcessor preProcessor;
    public transient ActionSet actionSet;

    public FrameHistory currentFrameHistory;
    public int next = 0;
    public FrameExperience[] experiences;
    public int size = 0;

    public boolean alwaysIncludeMostRecent;

    int maxHistoryLength; // the history size

    public FrameExperienceMemory(int size, int maxHistoryLength, PreProcessor preProcessor, ActionSet actionSet) {
        this(size, maxHistoryLength, preProcessor, actionSet, false);
    }

    public FrameExperienceMemory(int size, int maxHistoryLength, PreProcessor preProcessor, ActionSet actionSet, boolean alwaysIncludeMostRecent) {
        if(size < 1){
            throw new RuntimeException("FixedSizeMemory requires memory size > 0; was request size of " + size);
        }
        this.alwaysIncludeMostRecent = alwaysIncludeMostRecent;
        this.experiences = new FrameExperience[size];

        this.currentFrameHistory = new FrameHistory(0, 0);
        this.maxHistoryLength = maxHistoryLength;

        this.preProcessor = preProcessor;
        this.actionSet = actionSet;

        long outputSize = preProcessor.outputSize();

        // Create the frame history data size to be totalHistorySize + a padding on both sides of n - 1
        long paddingSize = (this.maxHistoryLength - 1) * outputSize;
        frameMemory = (new BytePointer(size * outputSize + 2 * paddingSize)).zero();
    }

    @Override
    public void vectorizeState(State state, FloatPointer input) {
        FrameHistory frameHistory = (FrameHistory) state;

        long frameSize = preProcessor.outputSize();
        long index = frameHistory.index;
        int historyLength = frameHistory.historyLength;

        long pos = input.position();
        input.limit(pos + maxHistoryLength * frameSize);

        // Fill unused frames with 0s
        if (historyLength < maxHistoryLength) {
            if (historyLength > 0) {
                input.limit(pos + (maxHistoryLength - historyLength)*frameSize).zero();
                input.limit(pos + maxHistoryLength * frameSize);
            } else {
                input.zero();
                return;
            }
        }

        // Convert compressed frameHistory data to CNN input
        preProcessor.convertDataToInput(
                this.frameMemory.position(index - (historyLength - 1)*frameSize),
                input.position(pos + (maxHistoryLength - historyLength)*frameSize),
                historyLength);
        input.position(pos);
    }

    @Override
    /** Assumes the input state is the most recently added state to the history **/
    public State mapState(State s) {
        return currentFrameHistory;
    }

    @Override
    public void addExperience(EnvironmentOutcome eo) {
        FrameHistory o = currentFrameHistory;
        FrameHistory op = addFrame(((ALEState)eo.op).getScreen());
        currentFrameHistory = op;

        experiences[next] = new FrameExperience(o, actionSet.map(eo.a), op, eo.r, eo.terminated);
        next = (next+1) % experiences.length;
        size = Math.min(size+1, experiences.length);
    }

    protected FrameHistory addFrame(Mat screenMat) {
        long outputSize = preProcessor.outputSize();
        long frameHistoryDataSize = frameMemory.capacity();
        long paddingSize = (maxHistoryLength - 1) * outputSize;

        // Find new index
        long newIndex = currentFrameHistory.index + outputSize;
        if (newIndex >= frameHistoryDataSize) {
            // Copy the buffer to the start of the history
            BytePointer frameHistoryCopy = new BytePointer(frameMemory);
            frameMemory.position(0).limit(paddingSize).put(frameHistoryCopy.position(frameHistoryDataSize - paddingSize));
            frameMemory.limit(frameMemory.capacity());

            newIndex = paddingSize;
        }

        // Increment length if smaller than n
        int newHistoryLength = currentFrameHistory.historyLength >= maxHistoryLength ?
                maxHistoryLength : currentFrameHistory.historyLength + 1;

        // Process the new screen and place it in the memory
        preProcessor.convertScreenToData(screenMat, frameMemory.position(newIndex));

        // Create new frame
        return new FrameHistory(newIndex, newHistoryLength);
    }

    @Override
    public List<EnvironmentOutcome> sampleExperiences(int n) {
        List<FrameExperience> samples = sampleFrameExperiences(n);

        List<EnvironmentOutcome> sampleOutcomes = new ArrayList<>(samples.size());
        for (FrameExperience exp : samples) {
            sampleOutcomes.add(new EnvironmentOutcome(exp.o, actionSet.get(exp.a), exp.op, exp.r, exp.terminated));
        }

        return sampleOutcomes;
    }

    public List<FrameExperience> sampleFrameExperiences(int n) {
        List<FrameExperience> samples;

        if(this.size == 0){
            return new ArrayList<>();
        }

        if(this.alwaysIncludeMostRecent){
            n--;
        }

        if(this.size < n){
            samples = new ArrayList<>(this.size);
            for(int i = 0; i < this.size; i++){
                samples.add(this.experiences[i]);
            }
            return samples;
        }
        else{
            samples = new ArrayList<>(Math.max(n, 1));
            Random r = RandomFactory.getMapped(0);
            for(int i = 0; i < n; i++) {
                int sind = r.nextInt(this.size);
                samples.add(this.experiences[sind]);
            }
        }
        if(this.alwaysIncludeMostRecent){
            FrameExperience eo;
            if(next > 0) {
                eo = this.experiences[next - 1];
            }
            else if(size > 0){
                eo = this.experiences[this.experiences.length-1];
            }
            else{
                throw new RuntimeException("FixedSizeMemory getting most recent fails because memory is size 0.");
            }
            samples.add(eo);
        }

        return samples;
    }

    @Override
    public void resetMemory() {
        this.size = 0;
        this.next = 0;
        this.currentFrameHistory = new FrameHistory(0, 0);
    }

    @Override
    public void saveMemory(String filePrefix) {

        String frameHistoryFilename = filePrefix + ".framehist";
        String frameExperienceFilename = filePrefix + ".ser";

        try (ObjectOutputStream objOut = new ObjectOutputStream(new FileOutputStream(frameExperienceFilename));
             FileOutputStream historyOut = new FileOutputStream(frameHistoryFilename)) {

            objOut.writeObject(this);

            // write frame history
            long pos = 0;
            byte[] buffer = new byte[10000000];
            this.frameMemory.get(buffer);
            int numRead;
            while (pos < frameMemory.limit()) {
                numRead = (int)Math.min(buffer.length, frameMemory.limit() - pos);
                frameMemory.position(pos).get(buffer, 0, numRead);
                pos += numRead;

                historyOut.write(buffer, 0, numRead);
            }

        } catch (IOException e) {
            System.out.println("Unable to save experience memory");
            e.printStackTrace();
            return;
        }
    }

    @Override
    public void loadMemory(String filePrefix) {
        String frameHistoryFilename = filePrefix + ".framehist";
        String frameExperienceFilename = filePrefix + ".ser";

        try (ObjectInputStream objIn = new ObjectInputStream(new FileInputStream(frameExperienceFilename));
             FileInputStream historyIn = new FileInputStream(frameHistoryFilename)) {

            // load object
            FrameExperienceMemory experienceMemory = (FrameExperienceMemory) objIn.readObject();
            this.currentFrameHistory = experienceMemory.currentFrameHistory;
            this.next = experienceMemory.next;
            this.size = experienceMemory.size;
            this.experiences = experienceMemory.experiences;


            // load frame history
            long pos = 0;
            byte[] buffer = new byte[10000000];
            int numRead;
            while ((numRead = historyIn.read(buffer)) != -1) {
                this.frameMemory.position(pos).put(buffer, 0, numRead);
                pos += numRead;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }
}
