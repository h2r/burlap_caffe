package edu.brown.cs.burlap.experiencereplay;

import burlap.mdp.core.state.State;

import java.io.Serializable;
import java.util.List;

/**
 * Created by MelRod on 5/27/16.
 */
public class FrameHistory implements State, Serializable {

    public long index;
    public int historyLength;


    FrameHistory(long index, int historyLength) {
        this.index = index;
        this.historyLength = historyLength;
    }

    @Override
    public List<Object> variableKeys() {
        return null;
    }

    @Override
    public Object get(Object variableKey) {
        return null;
    }

    @Override
    public State copy() {
        return null;
    }
}