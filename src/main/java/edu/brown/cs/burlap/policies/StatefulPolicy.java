package edu.brown.cs.burlap.policies;

import burlap.behavior.policy.Policy;

/**
 * An interface wrapper over Policy for loading information needed by the policy if restarting from some point in training.
 * For example, AnnealedEpsilonGreedy needs to know the current epsilon.
 *
 * @author Melrose Roderick.
 */
public interface StatefulPolicy extends Policy {
    void loadStateAt(int steps);
}
