package edu.brown.cs.burlap.experiencereplay;

import burlap.mdp.singleagent.environment.EnvironmentOutcome;

import java.util.List;

/**
 * @author James MacGlashan.
 */
public interface ExperienceMemory {
	void addExperience(EnvironmentOutcome eo);
	List<EnvironmentOutcome> sampleExperiences(int n);
	void resetMemory();

	void saveMemory(String filePrefix);
	void loadMemory(String filePrefix);
}
