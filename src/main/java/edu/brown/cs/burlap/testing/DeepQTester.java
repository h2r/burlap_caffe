package edu.brown.cs.burlap.testing;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.learning.experiencereplay.ExperienceMemory;
import burlap.behavior.singleagent.learning.experiencereplay.FixedSizeMemory;
import burlap.behavior.singleagent.options.EnvironmentOptionOutcome;
import burlap.mdp.auxiliary.StateMapping;
import burlap.mdp.auxiliary.common.ShallowIdentityStateMapping;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.burlap.vfa.DQN;

import java.util.List;

/**
 * A Tester for a DQN, which stores the frames in an ExperienceMemory
 * so information from previous states can be used if needed.
 *
 * @author Melrose Roderick.
 */
public class DeepQTester implements Tester {

    /**
     * The test policy
     */
    protected Policy policy;

    /**
     * The state mapping to convert between states
     */
    protected StateMapping stateMapping;

    /**
     * The experiences memory used for updating Q-values
     */
    protected ExperienceMemory memory;

    public DeepQTester(Policy policy, ExperienceMemory memory, StateMapping stateMapping) {
        this.policy = policy;
        this.memory = memory;
        this.stateMapping = stateMapping;
    }

    @Override
    public Episode runTestEpisode(Environment env, int maxSteps) {

        State initialState = env.currentObservation();
        Episode e = new Episode(initialState);


        int eStepCounter = 0;
        while(!env.isInTerminalState() && (eStepCounter < maxSteps || maxSteps == -1)){

            //check state
            State curState = stateMapping.mapState(env.currentObservation());

            //select action
            Action a = this.policy.action(curState);

            //take action
            EnvironmentOutcome eo = env.executeAction(a);

            //save outcome in memory
            this.memory.addExperience(eo);

            //record transition and manage option case
            int stepInc = eo instanceof EnvironmentOptionOutcome ? ((EnvironmentOptionOutcome)eo).numSteps() : 1;
            eStepCounter += stepInc;
            e.transition(a, eo.op, eo.r);

        }

        return e;
    }
}
