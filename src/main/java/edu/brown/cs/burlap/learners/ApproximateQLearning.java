package edu.brown.cs.burlap.learners;

import burlap.behavior.functionapproximation.ParametricFunction;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.MDPSolver;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.options.EnvironmentOptionOutcome;
import burlap.behavior.valuefunction.QProvider;
import burlap.behavior.valuefunction.QValue;
import burlap.mdp.auxiliary.StateMapping;
import burlap.mdp.auxiliary.common.ShallowIdentityStateMapping;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.burlap.experiencereplay.ExperienceMemory;

import java.util.ArrayList;
import java.util.List;

/**
 * @author James MacGlashan.
 */
public abstract class ApproximateQLearning extends MDPSolver implements LearningAgent, QProvider {


	/**
	 * The value function approximation used for Q-values.
	 */
	protected ParametricFunction.ParametricStateActionFunction vfa;

	/**
	 * The stale value function approximation used as the target toward which Q-values are updated.
	 */
	protected ParametricFunction.ParametricStateActionFunction staleVfa;

	/**
	 * The number of time steps until the stale VFA used as the target is updated to the newest Q-function estimate
	 */
	protected int staleDuration = 1;


	/**
	 * The number of learners steps that have been made since the stale function was last updated
	 */
	protected int stepsSinceStale = 0;



	/**
	 * The experiences memory used for updating Q-values
	 */
	protected ExperienceMemory memory;


	/**
	 * The state mapping to convert between states
	 */
	protected StateMapping stateMapping;


	/**
	 * The number of experiences to use for learners
	 */
	protected int numReplay = 1;

	/**
	 * The learners policy to use. Typically these will be policies that link back to this object so that they change as the Q-value estimate change.
	 */
	protected Policy learningPolicy;


	/**
	 * Whether options should be decomposed into actions in the returned {@link Episode} objects.
	 */
	protected boolean shouldDecomposeOptions = true;


	/**
	 * Whether decomposed options should have their primitive actions annotated with the options name in the returned {@link Episode} objects.
	 */
	protected boolean shouldAnnotateOptions = true;


	/**
	 * The total number of learners steps that have taken place
	 */
	protected int totalSteps = 0;
	protected int totalEpisodes = 0;


	public ApproximateQLearning(SADomain domain, double gamma, ParametricFunction.ParametricStateActionFunction vfa) {
		this(domain, gamma, vfa, new ShallowIdentityStateMapping());
	}
	public ApproximateQLearning(SADomain domain, double gamma, ParametricFunction.ParametricStateActionFunction vfa, StateMapping stateMapping) {
		this.vfa = vfa;
		this.staleVfa = vfa;
		this.learningPolicy = new EpsilonGreedy(this, 0.1);
		this.stateMapping = stateMapping;

		this.solverInit(domain, gamma, null);
	}

	/**
	 * Sets which policy this agent should use for learners.
	 * @param p the policy to use for learners.
	 */
	public void setLearningPolicy(Policy p){
		this.learningPolicy = p;
	}


	public void setExperienceReplay(ExperienceMemory memory, int numReplay){
		this.memory = memory;
		this.numReplay = numReplay;
	}

	public void useStaleTarget(int staleDuration){
		if(this.staleDuration <= 1 && staleDuration > 1){
			this.staleVfa = (ParametricFunction.ParametricStateActionFunction)this.vfa.copy();
		}
		if(this.staleDuration > 1 && staleDuration <= 1){
			this.staleVfa = this.vfa;
		}
		this.staleDuration = staleDuration;
	}


	@Override
	public Episode runLearningEpisode(Environment env) {
		return this.runLearningEpisode(env, -1);
	}

	@Override
	public Episode runLearningEpisode(Environment env, int maxSteps) {

		State initialState = env.currentObservation();
		Episode e = new Episode(initialState);


		int eStepCounter = 0;
		while(!env.isInTerminalState() && (eStepCounter < maxSteps || maxSteps == -1)){

			//check state
			State curState = stateMapping.mapState(env.currentObservation());

			//select action
			Action a = this.learningPolicy.action(curState);

			//take action
			EnvironmentOutcome eo = env.executeAction(a);

			//save outcome in memory
			this.memory.addExperience(eo);

			//record transition and manage option case
			int stepInc = eo instanceof EnvironmentOptionOutcome ? ((EnvironmentOptionOutcome)eo).numSteps() : 1;
			eStepCounter += stepInc;
			this.totalSteps += stepInc;
			e.transition(a, eo.op, eo.r);

			//perform learners
			List<EnvironmentOutcome> samples = this.memory.sampleExperiences(this.numReplay);
			this.updateQFunction(samples);

			//update stale function
			this.stepsSinceStale++;
			if(this.stepsSinceStale >= this.staleDuration){
				this.updateStaleFunction();
			}

		}

		this.totalEpisodes++;
		return e;
	}

	@Override
	public void resetSolver() {
		this.vfa.resetParameters();
		this.memory.resetMemory();
		this.totalSteps = 0;
		this.totalEpisodes = 0;
	}

	@Override
	public List<QValue> qValues(State s) {
		List<Action> actions = this.applicableActions(s);
		List<QValue> qs = new ArrayList<QValue>(actions.size());
		for(Action a : actions){
			QValue q = new QValue(s, a, this.qValue(s, a));
			qs.add(q);
		}
		return qs;
	}

	@Override
	public double qValue(State s, Action a) {
		return this.vfa.evaluate(s, a);
	}

	@Override
	public double value(State s) {
		List<QValue> qs = this.qValues(s);
		double max = Double.NEGATIVE_INFINITY;
		for(QValue q : qs){
			max = Math.max(max, q.q);
		}
		return max;
	}


	public List<QValue> getStaleQs(State s) {
		List<Action> actions = this.applicableActions(s);
		List<QValue> qs = new ArrayList<QValue>(actions.size());
		for(Action a : actions){
			QValue q = this.getStaleQ(s, a);
			qs.add(q);
		}
		return qs;
	}

	public QValue getStaleQ(State s, Action a) {
		double qv = this.staleVfa.evaluate(s, a);
		return new QValue(s, a, qv);
	}

	public double staleValue(State s) {
		List<QValue> qs = this.getStaleQs(s);
		double max = Double.NEGATIVE_INFINITY;
		for(QValue q : qs){
			max = Math.max(max, q.q);
		}
		return max;
	}

	public void updateStaleFunction(){
		if(this.staleDuration > 1){
			this.staleVfa = (ParametricFunction.ParametricStateActionFunction)this.vfa.copy();
		}
		else{
			this.staleVfa = this.vfa;
		}
		this.stepsSinceStale = 0;
	}

	public void resumeFrom(int stepNumber, int episodeNumber) {
		totalSteps = stepNumber;
		totalEpisodes = episodeNumber;
		updateStaleFunction();
	}

	public abstract void updateQFunction(List<EnvironmentOutcome> samples);
	public abstract void saveLearningState(String filePrefix);
	public abstract void loadLearningState(String filePrefix);
}
