package edu.brown.cs.burlap.learners;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.learning.tdmethods.vfa.ApproximateQLearning;
import burlap.mdp.auxiliary.StateMapping;
import burlap.mdp.auxiliary.common.ShallowIdentityStateMapping;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.burlap.experiencereplay.SavableExperienceMemory;
import edu.brown.cs.burlap.policies.StatefulPolicy;
import edu.brown.cs.burlap.vfa.DQN;

import java.io.*;
import java.util.HashMap;
import java.util.List;

/**
 * Created by MelRod on 5/24/16.
 */
public class DeepQLearner extends ApproximateQLearning {

    public int replayStartSize;
    public boolean runningRandomPolicy;

    public Policy trainingPolicy;

    public DeepQLearner(SADomain domain, double gamma, int replayStartSize, Policy policy, DQN vfa) {
        this(domain, gamma, replayStartSize, policy, vfa, new ShallowIdentityStateMapping());
    }
    public DeepQLearner(SADomain domain, double gamma, int replayStartSize, Policy policy, DQN vfa, StateMapping stateMapping) {
        super(domain, gamma, vfa, stateMapping);

        if (replayStartSize > 0) {
            System.out.println(String.format("Starting with random policy for %d frames", replayStartSize));

            this.replayStartSize = replayStartSize;
            this.trainingPolicy = policy;
            setLearningPolicy(new RandomPolicy(domain));
            runningRandomPolicy = true;
        } else {
            setLearningPolicy(policy);

            runningRandomPolicy = false;
        }
    }

    @Override
    public void updateQFunction(List<EnvironmentOutcome> samples) {

        // fill up experience replay
        if (runningRandomPolicy) {
            if (totalSteps >= replayStartSize) {
                System.out.println("Replay sufficiently filled. Beginning training...");

                setLearningPolicy(trainingPolicy);
                runningRandomPolicy = false;
            }

            return;
        }

        ((DQN)vfa).updateQFunction(samples, (DQN)staleVfa);
    }

    @Override
    public void updateStaleFunction() {
        if (this.staleDuration > 1) {
            ((DQN)this.staleVfa).updateParamsToMatch((DQN)this.vfa);
        } else {
            this.staleVfa = this.vfa;
        }
        this.stepsSinceStale = 1;
    }

    public void saveLearningState(String filePrefix) {
        // Save meta data
        String dataFilename = filePrefix + "_learner.data";
        HashMap<String, Object> data = new HashMap<>();
        data.put("totalSteps", totalSteps);
        data.put("totalEpisodes", totalEpisodes);
        try (ObjectOutputStream objOut = new ObjectOutputStream(new FileOutputStream(dataFilename))) {
            objOut.writeObject(data);
        } catch (IOException e) {
            System.out.println("Unable to save learning state");
            e.printStackTrace();
            return;
        }

        // Save experience memory
        if (memory instanceof SavableExperienceMemory) {
            ((SavableExperienceMemory) memory).saveMemory(filePrefix);
        }

        // Save Caffe data
        ((DQN) vfa).saveLearningState(filePrefix);
    }

    public void loadLearningState(String filePrefix) {

        // Load meta-data
        String dataFilename = filePrefix + "_learner.data";
        try (ObjectInputStream objIn = new ObjectInputStream(new FileInputStream(dataFilename))) {
            HashMap<String, Object> data = (HashMap<String, Object>) objIn.readObject();

            resumeFrom((Integer)data.get("totalSteps"), (Integer)data.get("totalEpisodes"));
        } catch (FileNotFoundException e) {
            System.out.println("No learning state found for specified file name");
            e.printStackTrace();
            return;
        } catch (IOException e) {
            System.out.println("Unable to load learning state");
            e.printStackTrace();
            return;
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            return;
        }

        // Load experience memory
        if (memory instanceof SavableExperienceMemory) {
            ((SavableExperienceMemory) memory).loadMemory(filePrefix);
        }

        // Load Caffe solver state
        ((DQN)vfa).loadLearningState(filePrefix);

        if (runningRandomPolicy) {
            if (totalSteps >= replayStartSize) {
                System.out.println("Replay sufficiently filled. Beginning training...");

                setLearningPolicy(trainingPolicy);
                runningRandomPolicy = false;
            }
        }

        // If the policy depends on the iteration (i.e. AnnealedEpsilonGreedy) then load the state for the current step
        if (this.learningPolicy instanceof StatefulPolicy) {
            ((StatefulPolicy)this.learningPolicy).loadStateAt(this.totalSteps);
        }
    }
}
