package edu.brown.cs.burlap.experiencereplay;

import burlap.behavior.singleagent.learning.experiencereplay.ExperienceMemory;

/**
 * An interface on top of {@code ExperienceMemory} for saving a snapshot of the memory in case of interrupts.
 *
 * @author Melrose Roderick.
 */
public interface SavableExperienceMemory extends ExperienceMemory {
    void saveMemory(String filePrefix);
    void loadMemory(String filePrefix);
}
