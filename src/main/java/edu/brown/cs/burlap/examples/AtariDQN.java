package edu.brown.cs.burlap.examples;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.SolverDerivedPolicy;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import edu.brown.cs.burlap.ALEDomainGenerator;
import edu.brown.cs.burlap.ALEEnvironment;
import edu.brown.cs.burlap.action.ActionSet;
import edu.brown.cs.burlap.experiencereplay.FrameExperienceMemory;
import edu.brown.cs.burlap.gui.ALEVisualExplorer;
import edu.brown.cs.burlap.gui.ALEVisualizer;
import edu.brown.cs.burlap.io.PoolingMethod;
import edu.brown.cs.burlap.learners.DeepQLearner;
import edu.brown.cs.burlap.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.burlap.preprocess.ALEPreProcessor;
import edu.brown.cs.burlap.testing.DeepQTester;
import edu.brown.cs.burlap.testing.SimpleTester;
import edu.brown.cs.burlap.testing.Tester;
import edu.brown.cs.burlap.vfa.DQN;
import org.bytedeco.javacpp.Loader;

import static org.bytedeco.javacpp.caffe.Caffe;

/**
 * Created by MelRod on 5/31/16.
 */
public class AtariDQN extends TrainingHelper {

    // TODO: set to true if you download our version of ALE and want to replicate the Deepmind results
    static final boolean TERMINATE_ON_END_LIFE = false;

    protected FrameExperienceMemory trainingMemory;
    protected FrameExperienceMemory testMemory;

    public AtariDQN(DeepQLearner learner, DeepQTester tester, DQN vfa, ActionSet actionSet, Environment env,
                    FrameExperienceMemory trainingMemory,
                    FrameExperienceMemory testMemory) {
        super(learner, tester, vfa, actionSet, env);

        this.trainingMemory = trainingMemory;
        this.testMemory = testMemory;
    }

    @Override
    public void prepareForTraining() {
        if (TERMINATE_ON_END_LIFE) {
            ((ALEEnvironment) env).terminateOnEndLife = true;
        }

        vfa.stateConverter = trainingMemory;
    }

    @Override
    public void prepareForTesting() {
        if (TERMINATE_ON_END_LIFE) {
            ((ALEEnvironment) env).terminateOnEndLife = false;
        }

        vfa.stateConverter = testMemory;
    }

    public static void main(String[] args) {

        // Learning constants defined in the Deep-Mind Nature paper
        // (http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
        int experienceMemoryLength = 1000000;
        int maxHistoryLength = 4;
        int staleUpdateFreq = 10000;
        double gamma = 0.99;
        int frameSkip = 4;
        int updateFreq = 4;
        double epsilonStart = 1;
        double epsilonEnd = 0.1;
        int epsilonAnnealDuration = 1000000;
        int replayStartSize = 50000;
        int totalTrainingSteps = 50000000;
        double testEpsilon = 0.05;

        // Testing and recording constants
        int testInterval = 250000;
        int numTestEpisodes = 50;
        int maxEpisodeSteps = 100000;
        int snapshotInterval = 1000000;
        String snapshotDirectory = "snapshots/experiment1";
        String resultsDirectory = "results/experiment1";

        // ALE Paths
        // TODO: Set to appropriate paths for your machine
        String alePath = "/path/to/atari/executable";
        String romPath = "/path/to/atari/rom/file";

        // Caffe solver file
        String solverFile = "example_models/atari_dqn_solver.prototxt";

        // Load Caffe
        Loader.load(Caffe.class);

        // Create the domain
        ALEDomainGenerator domGen = new ALEDomainGenerator(ALEDomainGenerator.saActionSet());
        SADomain domain = domGen.generateDomain();

        // Create the ALEEnvironment and visualizer
        ALEEnvironment env = new ALEEnvironment(alePath, romPath, frameSkip, PoolingMethod.POOLING_METHOD_MAX);
        ALEVisualExplorer exp = new ALEVisualExplorer(domain, env, ALEVisualizer.create());
        exp.initGUI();
        exp.startLiveStatePolling(1000/60);

        // Setup the ActionSet from the ALEDomain to use the ALEActions
        ActionSet actionSet = new ActionSet(domain);

        // Setup the training and test memory
        FrameExperienceMemory trainingExperienceMemory =
                new FrameExperienceMemory(experienceMemoryLength, maxHistoryLength, new ALEPreProcessor(), actionSet);
        // The size of the test memory is arbitrary but should be significantly greater than 1 to minimize copying
        FrameExperienceMemory testExperienceMemory =
                new FrameExperienceMemory(10000, maxHistoryLength, new ALEPreProcessor(), actionSet);


        // Initialize the DQN with the solver file.
        // NOTE: this Caffe architecture is made for 3 actions (the number of actions in Pong)
        DQN dqn = new DQN(solverFile, actionSet, trainingExperienceMemory, gamma);

        // Create the policies
        SolverDerivedPolicy learningPolicy =
                new AnnealedEpsilonGreedy(epsilonStart, epsilonEnd, epsilonAnnealDuration);
        SolverDerivedPolicy testPolicy = new EpsilonGreedy(testEpsilon);

        // Setup the learner
        DeepQLearner deepQLearner = new DeepQLearner(domain, gamma, replayStartSize, learningPolicy, dqn, trainingExperienceMemory);
        deepQLearner.setExperienceReplay(trainingExperienceMemory, dqn.batchSize);
        deepQLearner.useStaleTarget(staleUpdateFreq);
        deepQLearner.setUpdateFreq(updateFreq);

        // Setup the tester
        DeepQTester tester = new DeepQTester(testPolicy, testExperienceMemory, testExperienceMemory);

        // Set the QProvider for the policies
        learningPolicy.setSolver(deepQLearner);
        testPolicy.setSolver(deepQLearner);

        // Setup helper
        TrainingHelper helper =
                new AtariDQN(deepQLearner, tester, dqn, actionSet, env, trainingExperienceMemory, testExperienceMemory);
        helper.setTotalTrainingSteps(totalTrainingSteps);
        helper.setTestInterval(testInterval);
        helper.setNumTestEpisodes(numTestEpisodes);
        helper.setMaxEpisodeSteps(maxEpisodeSteps);
        helper.enableSnapshots(snapshotDirectory, snapshotInterval);
        helper.recordResultsTo(resultsDirectory);

        // Load learning state if resuming
        //helper.loadLearningState("networks/dqn/pong");

        // Run helper
        helper.run();
    }
}
