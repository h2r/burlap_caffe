package edu.brown.cs.burlap.testing;

import burlap.behavior.singleagent.Episode;
import burlap.mdp.singleagent.environment.Environment;

/**
 * Created by maroderi on 8/31/16.
 */
public interface Tester {
    Episode runTestEpisode(Environment env, int maxSteps);
}
