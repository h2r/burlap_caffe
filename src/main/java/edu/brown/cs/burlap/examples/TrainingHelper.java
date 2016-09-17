package edu.brown.cs.burlap.examples;

import burlap.behavior.singleagent.Episode;
import burlap.mdp.singleagent.environment.Environment;
import edu.brown.cs.burlap.action.ActionSet;
import edu.brown.cs.burlap.learners.DeepQLearner;
import edu.brown.cs.burlap.testing.Tester;
import edu.brown.cs.burlap.vfa.DQN;

import java.io.*;

/**
 * A class to coordinate all the steps for training and testing a DQN on a given Domain.
 *
 * @author Melrose Roderick.
 */
public class TrainingHelper {

    protected DeepQLearner learner;
    protected Tester tester;
    protected DQN vfa;

    protected Environment env;

    protected ActionSet actionSet;

    protected int maxEpisodeSteps = -1;
    protected int totalTrainingSteps = 10000000;

    protected int testInterval = 100000;
    protected int numTestEpisodes = 10;

    protected String snapshotPrefix;
    protected int snapshotInterval = -1;

    protected int stepCounter;
    protected int episodeCounter;

    protected double highestAverageReward = Double.NEGATIVE_INFINITY;
    protected PrintStream testOutput;
    protected String resultsPrefix;


    public TrainingHelper(DeepQLearner learner, Tester tester, DQN vfa, ActionSet actionSet, Environment env) {
        this.learner = learner;
        this.vfa = vfa;
        this.tester = tester;
        this.env = env;
        this.actionSet = actionSet;

        this.stepCounter = 0;
        this.episodeCounter = 0;
    }

    public void prepareForTraining() {}
    public void prepareForTesting() {}

    public void setTotalTrainingSteps(int n) {
        totalTrainingSteps = n;
    }

    public void setNumTestEpisodes(int n) {
        numTestEpisodes = n;
    }

    public void setTestInterval(int i) {
        testInterval = i;
    }

    public void setMaxEpisodeSteps(int f) {
        maxEpisodeSteps = f;
    }

    public void enableSnapshots(String snapshotPrefix, int snapshotInterval) {
        File dir = new File(snapshotPrefix);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new RuntimeException(String.format("Could not create the directory: %s", snapshotPrefix));
        }

        this.snapshotPrefix = snapshotPrefix;
        this.snapshotInterval = snapshotInterval;
    }

    public void recordResultsTo(String resultsPrefix) {
        File dir = new File(resultsPrefix);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new RuntimeException(String.format("Could not create the directory: %s", resultsPrefix));
        }

        this.resultsPrefix = resultsPrefix;

        try {
            String fileName = new File(resultsPrefix, "testResults").toString();
            testOutput = new PrintStream(new BufferedOutputStream(new FileOutputStream(fileName)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();

            throw new RuntimeException(String.format("Can't open %s", resultsPrefix));
        }
    }

    public void run() {

        int testCountDown = testInterval;
        int snapshotCountDown = snapshotInterval;

        while (stepCounter < totalTrainingSteps) {
            System.out.println(String.format("Training Episode %d at step %d", episodeCounter, stepCounter));

            prepareForTraining();
            env.resetEnvironment();

            long startTime = System.currentTimeMillis();
            Episode ea = learner.runLearningEpisode(env, Math.min(totalTrainingSteps - stepCounter, maxEpisodeSteps));
            long endTime = System.currentTimeMillis();
            double timeInterval = (endTime - startTime)/1000.0;

            double totalReward = 0;
            for (double r : ea.rewardSequence) {
                totalReward += r;
            }
            System.out.println(String.format("Episode reward: %.2f -- %.1f steps/sec", totalReward, ea.numTimeSteps()/timeInterval));
            System.out.println();

            stepCounter += ea.numTimeSteps();
            episodeCounter++;
            if (snapshotPrefix != null) {
                snapshotCountDown -= ea.numTimeSteps();
                if (snapshotCountDown <= 0) {
                    saveLearningState(snapshotPrefix);
                    snapshotCountDown += snapshotInterval;
                }
            }

            testCountDown -= ea.numTimeSteps();
            if (testCountDown <= 0) {
                runTestSet();
                testCountDown += testInterval;
            }
        }

        if (testOutput != null) {
            testOutput.printf("Final best: %.2f\n", highestAverageReward);
            testOutput.flush();
        }

        System.out.println("Done Training!");
    }

    public void runTestSet() {

        // Change any learning variables to test values (i.e. experience memory)
        prepareForTesting();

        // Run the test policy on test episodes
        System.out.println("Running Test Set...");
        double totalTestReward = 0;
        for (int n = 1; n <= numTestEpisodes; n++) {
            env.resetEnvironment();
            Episode e = tester.runTestEpisode(env, maxEpisodeSteps);

            double totalReward = 0;
            for (double reward : e.rewardSequence) {
                totalReward += reward;
            }

            System.out.println(String.format("%d: Reward = %.2f", n, totalReward));
            totalTestReward += totalReward;
        }

        double averageReward = totalTestReward/numTestEpisodes;
        if (averageReward > highestAverageReward) {
            if (resultsPrefix != null) {
                vfa.snapshot(new File(resultsPrefix, "best_net.caffemodel").toString(),  null);
            }
            highestAverageReward = averageReward;
        }

        System.out.println(String.format("Average Test Reward: %.2f -- highest: %.2f", totalTestReward/numTestEpisodes, highestAverageReward));
        System.out.println();

        if (testOutput != null) {
            testOutput.printf("Frame %d: %.2f\n", stepCounter, averageReward);
            testOutput.flush();
        }
    }

    public void saveLearningState(String filePrefix) {
        learner.saveLearningState(filePrefix);
    }

    public void loadLearningState(String filePrefix) {
        learner.loadLearningState(filePrefix);
    }
}
