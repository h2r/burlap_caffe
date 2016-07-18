package edu.brown.cs.burlap.examples;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.UniformCostRF;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import edu.brown.cs.burlap.action.ActionSet;
import edu.brown.cs.burlap.experiencereplay.FixedSizeMemory;
import edu.brown.cs.burlap.learners.DeepQLearner;
import edu.brown.cs.burlap.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.burlap.vfa.DQN;
import edu.brown.cs.burlap.vfa.StateVectorizor;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.caffe;


/**
 * Created by MelRod on 5/29/16.
 */
public class GridWorldDQN {

    static final String SOLVER_FILE = "example_models/grid_world_dqn_solver.prototxt";

    static ActionSet actionSet = new ActionSet(new String[]{
            GridWorldDomain.ACTION_NORTH,
            GridWorldDomain.ACTION_SOUTH,
            GridWorldDomain.ACTION_WEST,
            GridWorldDomain.ACTION_EAST});

    public GridWorldDomain gwdg;
    public OOSADomain domain;
    public RewardFunction rf;
    public TerminalFunction tf;
    public StateConditionTest goalCondition;
    public State initialState;
    public HashableStateFactory hashingFactory;
    public Environment env;

    public DQN dqn;

    public GridWorldDQN(double gamma) {

        //create the domain
        gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();
        rf = new UniformCostRF();
        tf = new SinglePFTF(PropositionalFunction.findPF(gwdg.generatePfs(), GridWorldDomain.PF_AT_LOCATION));
        gwdg.setRf(rf);
        gwdg.setTf(tf);
        domain = gwdg.generateDomain();

        goalCondition = new TFGoalCondition(tf);

        //set up the initial state of the task
        initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));

        //set up the state hashing system for tabular algorithms
        hashingFactory = new SimpleHashableStateFactory();

        //set up the environment for learners algorithms
        env = new SimulatedEnvironment(domain, initialState);

        dqn = new DQN(SOLVER_FILE, actionSet, new NNGridStateConverter(), gamma);
    }

    public static void main(String args[]) {

        // Learning constants
        double gamma = 0.99;
        int replayStartSize = 50000;
        int memorySize = 1000000;
        double epsilonStart = 1;
        double epsilonEnd = 0.1;
        int epsilonAnnealDuration = 1000000;

        // Load Caffe
        Loader.load(caffe.Caffe.class);

        // Setup the network
        GridWorldDQN gridWorldDQN = new GridWorldDQN(gamma);

        // Create the policies
        Policy learningPolicy =
                new AnnealedEpsilonGreedy(gridWorldDQN.dqn, epsilonStart, epsilonEnd, epsilonAnnealDuration);
        Policy testPolicy = new EpsilonGreedy(gridWorldDQN.dqn, 0.05);

        // Setup the learner
        DeepQLearner deepQLearner =
                new DeepQLearner(gridWorldDQN.domain, gamma, replayStartSize, learningPolicy, gridWorldDQN.dqn);
        deepQLearner.setExperienceReplay(new FixedSizeMemory(memorySize), gridWorldDQN.dqn.batchSize);

        // Setup the visualizer
        VisualExplorer exp = new VisualExplorer(
                gridWorldDQN.domain, gridWorldDQN.env, GridWorldVisualizer.getVisualizer(gridWorldDQN.gwdg.getMap()));
        exp.initGUI();
        exp.startLiveStatePolling(33);

        // Setup helper
        TrainingHelper helper = new TrainingHelper.SimpleTrainer(
                deepQLearner, gridWorldDQN.dqn, testPolicy, actionSet, gridWorldDQN.env);
        helper.setTotalTrainingSteps(50000000);
        helper.setTestInterval(500000);
        helper.setNumTestEpisodes(5);
        helper.setMaxEpisodeSteps(10000);

        // Run helper
        helper.run();
    }

    class NNGridStateConverter implements StateVectorizor {

        @Override
        public void vectorizeState(State state, FloatPointer input) {
            GridWorldState gwState = (GridWorldState) state;

            int width = gwdg.getWidth();

            input.fill(0);

            ObjectInstance agent = gwState.object(GridWorldDomain.CLASS_AGENT);
            int x = (Integer)agent.get(GridWorldDomain.VAR_X);
            int y = (Integer)agent.get(GridWorldDomain.VAR_Y);

            input.put((long)(y*width + x), 1);
        }
    }
}
