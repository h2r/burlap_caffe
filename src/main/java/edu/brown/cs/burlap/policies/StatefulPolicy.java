package edu.brown.cs.burlap.policies;

import burlap.behavior.policy.Policy;

/**
 * Created by maroderi on 7/13/16.
 */
public interface StatefulPolicy extends Policy {
    void loadStateAt(int steps);
}
