package edu.brown.cs.burlap.vfa;

import burlap.mdp.core.state.State;
import org.bytedeco.javacpp.FloatPointer;

/**
 * Created by MelRod on 5/27/16.
 */
public interface StateVectorizor {

    /**
     * Converts a given state to a float vector
     *
     * @param state The state to convert.
     * @param input The float vector into which to put the state vector.
     */
    void vectorizeState(State state, FloatPointer input);
}
