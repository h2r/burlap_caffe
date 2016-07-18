package edu.brown.cs.burlap.experiencereplay;

import burlap.behavior.singleagent.learning.experiencereplay.ExperienceMemory;

/**
 * Created by maroderi on 7/18/16.
 */
public interface SavableExperienceMemory extends ExperienceMemory {
    void saveMemory(String filePrefix);
    void loadMemory(String filePrefix);
}
