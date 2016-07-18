package edu.brown.cs.burlap.examples;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.mdp.singleagent.environment.Environment;
import edu.brown.cs.burlap.action.ActionSet;
import edu.brown.cs.burlap.learners.DeepQLearner;
import edu.brown.cs.burlap.vfa.DQN;

/**
 * Created by MelRod on 5/28/16.
 */
public abstract class TrainingHelper {

    protected DeepQLearner learner;
    protected DQN vfa;
    protected Policy testPolicy;

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


    public TrainingHelper(DeepQLearner learner, DQN vfa, Policy testPolicy, ActionSet actionSet, Environment env) {
        this.learner = learner;
        this.vfa = vfa;
        this.testPolicy = testPolicy;
        this.env = env;
        this.actionSet = actionSet;

        this.stepCounter = 0;
        this.episodeCounter = 0;
    }

    public abstract void prepareForTraining();
    public abstract void prepareForTesting();

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
        this.snapshotPrefix = snapshotPrefix;
        this.snapshotInterval = snapshotInterval;
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

        System.out.println("Done Training!");
    }

    public void runTestSet() {

        // Change any learning variables to test values (i.e. experience memory)
        prepareForTesting();

        // Run the test policy on test episodes
        System.out.println("Running Test Set...");
        double totalTestReward = 0;
        for (int n = 1; n <= numTestEpisodes; n++) {
            Episode e = PolicyUtils.rollout(testPolicy, env, maxEpisodeSteps);

            double totalReward = 0;
            for (double reward : e.rewardSequence) {
                totalReward += reward;
            }

            System.out.println(String.format("%d: Reward = %.2f", n, totalReward));
            totalTestReward += totalReward;
        }

        System.out.println(String.format("Average Test Reward: %.2f", totalTestReward/numTestEpisodes));
        System.out.println();
    }

    public void saveLearningState(String filePrefix) {
        learner.saveLearningState(filePrefix);
    }

    public void loadLearningState(String filePrefix) {
        learner.loadLearningState(filePrefix);
    }


    public static class SimpleTrainer extends TrainingHelper {

        public SimpleTrainer(DeepQLearner learner, DQN vfa, Policy testPolicy, ActionSet actionSet, Environment env) {
            super(learner, vfa, testPolicy, actionSet, env);
        }
        @Override
        public void prepareForTraining() {}
        @Override
        public void prepareForTesting() {}
    }
}
