package edu.brown.cs.burlap.testing;

import burlap.behavior.singleagent.Episode;
import burlap.mdp.singleagent.environment.Environment;

/**
 * An interface for running test episodes in a specified Environment.
 *
 * @author Melrose Roderick.
 */
public interface Tester {
    Episode runTestEpisode(Environment env, int maxSteps);
}
