package edu.brown.cs.burlap.experiencereplay;

import java.io.Serializable;

/**
 * Created by maroderi on 6/29/16.
 */
public class FrameExperience implements Serializable {

    /** The ActionSet id of the action that was taken */
    public int a;

    /** The reward received */
    public double r;

    /** True if the experience resulted in a terminal state */
    public boolean terminated;

    /** The state from which the action was taken */
    public FrameHistory o;

    /** The state at which the agent arrived */
    public FrameHistory op;

    public FrameExperience(FrameHistory o, int a, FrameHistory op, double r, boolean terminated) {
        this.o = o;
        this.a = a;
        this.op = op;
        this.r = r;
        this.terminated = terminated;
    }
}
