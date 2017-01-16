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
    protected int totalTestSteps = 125000;

    protected String snapshotPrefix;
    protected int snapshotInterval = -1;

    protected int stepCounter;
    protected int episodeCounter;

    protected double highestAverageReward = Double.NEGATIVE_INFINITY;
    protected PrintStream testOutput;
    protected String resultsPrefix;

    /** If true, prints out episode information at the end of every episode */
    public boolean verbose = false;


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

    public void setTotalTestSteps(int n) {
        totalTestSteps = n;
    }

    public void setTestInterval(int i) {
        testInterval = i;
    }

    public void setMaxEpisodeSteps(int f) {
        maxEpisodeSteps = f;
    }

    public void enableSnapshots(String snapshotPrefix, int snapshotInterval) {
        File dir = new File(snapshotPrefix);
        File parent = dir.getParentFile();
        if (!parent.exists() && !parent.mkdirs()) {
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

        long trainingStart = System.currentTimeMillis();
        int trainingSteps = 0;
        while (stepCounter < totalTrainingSteps) {
            long epStartTime = 0;
            if (verbose) {
                System.out.println(String.format("Training Episode %d at step %d", episodeCounter, stepCounter));
                epStartTime = System.currentTimeMillis();
            }

            // Set variables needed for training
            prepareForTraining();
            env.resetEnvironment();

            // run learning episode
            Episode ea = learner.runLearningEpisode(env, Math.min(totalTrainingSteps - stepCounter, maxEpisodeSteps));

            // add up episode reward
            double totalReward = 0;
            for (double r : ea.rewardSequence) {
                totalReward += r;
            }

            if (verbose) {
                // output episode data
                long epEndTime = System.currentTimeMillis();
                double timeInterval = (epEndTime - epStartTime)/1000.0;

                System.out.println(String.format("Episode reward: %.2f -- %.1f steps/sec", totalReward, ea.numTimeSteps()/timeInterval));
                System.out.println();
            }

            // take snapshot every snapshotCountDown steps
            stepCounter += ea.numTimeSteps();
            trainingSteps += ea.numTimeSteps();
            episodeCounter++;
            if (snapshotPrefix != null) {
                snapshotCountDown -= ea.numTimeSteps();
                if (snapshotCountDown <= 0) {
                    saveLearningState(snapshotPrefix);
                    snapshotCountDown += snapshotInterval;
                }
            }

            // take test set every testCountDown steps
            testCountDown -= ea.numTimeSteps();
            if (testCountDown <= 0) {
                double trainingTimeInterval = (System.currentTimeMillis() - trainingStart)/1000.0;

                // run test set
                runTestSet();
                testCountDown += testInterval;

                // output training rate
                System.out.printf("Training rate: %.1f steps/sec\n\n",
                        testInterval/trainingTimeInterval);

                // restart training timer
                trainingStart = System.currentTimeMillis();
            }
        }

        if (testOutput != null) {
            testOutput.printf("Final best: %.2f\n", highestAverageReward);
            testOutput.flush();
        }

        System.out.println("Done Training!");
    }

    public void runTestSet() {

        long testStart = System.currentTimeMillis();
        int numSteps = 0;
        int numEpisodes = 0;

        // Change any learning variables to test values (i.e. experience memory)
        prepareForTesting();

        // Run the test policy on test episodes
        System.out.println("Running Test Set...");
        double totalTestReward = 0;
        while (true) {
            env.resetEnvironment();
            Episode e = tester.runTestEpisode(env, Math.min(maxEpisodeSteps, totalTestSteps - numSteps));

            double totalReward = 0;
            for (double reward : e.rewardSequence) {
                totalReward += reward;
            }

            if (verbose) {
                System.out.println(String.format("%d: Reward = %.2f, Steps = %d", numEpisodes, totalReward, numSteps));
            }

            numSteps += e.numTimeSteps();
            if (numSteps >= totalTestSteps) {
                if (numEpisodes == 0) {
                    totalTestReward = totalReward;
                    numEpisodes = 1;
                }
                break;
            }

            totalTestReward += totalReward;
            numEpisodes += 1;
        }

        double averageReward = totalTestReward/numEpisodes;
        if (averageReward > highestAverageReward) {
            if (resultsPrefix != null) {
                vfa.snapshot(new File(resultsPrefix, "best_net.caffemodel").toString(),  null);
            }
            highestAverageReward = averageReward;
        }

        double testTimeInterval = (System.currentTimeMillis() - testStart)/1000.0;
        System.out.printf("Average Test Reward: %.2f -- highest: %.2f, Test rate: %.1f\n\n", averageReward, highestAverageReward, numSteps/testTimeInterval);

        if (testOutput != null) {
            testOutput.printf("Frame %d: %.2f\n", stepCounter, averageReward);
            testOutput.flush();
        }
    }

    public void saveLearningState(String filePrefix) {
        System.out.print("Saving learning snapshot... ");
        learner.saveLearningState(filePrefix);
        System.out.println("Done");
    }

    public void loadLearningState(String filePrefix) {
        learner.loadLearningState(filePrefix);
    }
}
