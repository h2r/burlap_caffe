package edu.brown.cs.burlap.policies;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.valuefunction.QProvider;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

/**
 * Created by MelRod on 5/4/16.
 */
public class AnnealedEpsilonGreedy extends EpsilonGreedy implements StatefulPolicy {

    protected double epsilonStart;
    protected double epsilonEnd;
    protected double epsilonStep;
    protected int annealingTime;

    public AnnealedEpsilonGreedy(double epsilonStart, double epsilonEnd, int annealingTime) {
        this(null, epsilonStart, epsilonEnd, annealingTime);
    }
    public AnnealedEpsilonGreedy(QProvider planner, double epsilonStart, double epsilonEnd, int annealingTime) {
        super(planner, epsilonStart);

        this.epsilonStart = epsilonStart;
        this.epsilonEnd = epsilonEnd;
        this.epsilonStep = (epsilonEnd - epsilonStart)/annealingTime;
        this.annealingTime = annealingTime;
    }

    @Override
    public Action action(State s) {
        Action action = super.action(s);

        if (epsilon > epsilonEnd) {
            epsilon += epsilonStep;

            if (epsilon < epsilonEnd) {
                epsilon = epsilonEnd;
            }
        }

        return action;
    }

    @Override
    public void loadStateAt(int steps) {
        epsilon += epsilonStep * steps;
        if (epsilon < epsilonEnd) {
            epsilon = epsilonEnd;
        }
    }
}
