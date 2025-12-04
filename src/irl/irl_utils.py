import numpy as np

def compute_state_count(state_sequence,n_states,n_actions):
	tp = np.zeros([n_states,n_actions])

	for pair in np.arange(len(state_sequence)):
		s = state_sequence[pair][0]
		a = state_sequence[pair][1]
		tp[s, a] += 1
	return tp

def compute_tp(state_sequence,n_states,n_actions):
    """computes the transition probablities based on the trajectory state-action pair sequence

    Args:
        state_sequence (_type_): trajectory state-action pair sequence
        n_states (_type_): number of states
        n_actions (_type_): number of actions

    Returns:
        _type_: transition probablities in format of (state,action,next_state)
    """
    tp = np.zeros([n_states,n_actions,n_states])

    for pair in np.arange(len(state_sequence)-1):
        s = state_sequence[pair][0]
        a = state_sequence[pair][1]
        ns = state_sequence[pair+1][0]
        tp[s, a, ns] += 1

    for c in np.arange(len(tp)):
        row_sums = tp[c].sum(axis=1)
        # prevent division by zero by dividing by 1 for rows that sum to 0
        row_sums[row_sums == 0] = 1
        A = tp[c] / row_sums[:, None]
        tp[c] = A
    return tp

def legalise_tp(tp, legal_transitions):
    """
    changes tp entries which are all zeros such that they sum to 1 by dividing the probablity over all legal transitions    
    """
    legal_tp = tp.copy()
    for s in range(tp.shape[0]):
        for a in range(tp.shape[1]):
            for ns in range(tp.shape[2]):
                if np.sum(tp[s,a]) == 0:
                    legal_tp[s,a] = legal_transitions[s,a]/np.sum(legal_transitions[s,a])
    

    return legal_tp