package edu.brown.cs.burlap.testing;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.mdp.auxiliary.common.ShallowIdentityStateMapping;
import burlap.mdp.singleagent.environment.Environment;

/**
 * A simple implementation of Tester, which just runs the PolicyUtils.rollout.
 *
 * @author Melrose Roderick.
 */
public class SimpleTester implements Tester {

    /**
     * The test policy
     */
    Policy policy;

    public SimpleTester(Policy policy) {
        this.policy = policy;
    }

    @Override
    public Episode runTestEpisode(Environment env, int maxSteps) {
        return PolicyUtils.rollout(policy, env, maxSteps);
    }
}
