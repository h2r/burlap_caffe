package edu.brown.cs.burlap.action;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionUtils;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.singleagent.SADomain;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by MelRod on 5/28/16.
 */
public class ActionSet {

    protected Action[] actions;
    protected Map<String, Integer> actionMap;
    protected int size;

    public ActionSet(SADomain domain) {
        List<Action> actionList = ActionUtils.allApplicableActionsForTypes(domain.getActionTypes(), null);
        size = actionList.size();
        actions = new Action[size];
        actionList.toArray(actions);

        initActionMap();
    }
    public ActionSet(Action[] actions) {
        this.actions = actions;
        size = actions.length;

        initActionMap();
    }
    public ActionSet(String[] actionNames) {
        size = actionNames.length;
        actions = new Action[size];
        for (int i = 0; i < size; i++) {
            actions[i] = new SimpleAction(actionNames[i]);
        }

        initActionMap();
    }

    protected void initActionMap() {
        actionMap = new HashMap<>();
        for (int i = 0; i < actions.length; i++) {
            actionMap.put(actions[i].actionName(), i);
        }
    }

    public Action get(int i) {
        return actions[i];
    }

    public int map(String action) {
        return actionMap.get(action);
    }
    public int map(Action action) {
        return actionMap.get(action.actionName());
    }

    public int size() {
        return size;
    }
}