package edu.brown.cs.burlap.vfa;

import burlap.mdp.core.state.State;
import org.bytedeco.javacpp.FloatPointer;

/**
 * An interface for converting a state into a float vector to pass through Caffe.
 *
 * @author Melrose Roderick.
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
