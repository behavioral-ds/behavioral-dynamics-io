import numpy as np

state_map_no_init = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4,
    5:5,
    6:6,
    7:7,
    8:4,
    9:5,
    10:6,
    11:7
}
state_map_no_agree = {
    0:0,
    1:0,
    2:0,
    3:1,
    4:2,
    5:3,
    6:3,
    7:3,
    8:4,
    9:5,
    10:5,
    11:5
}
action_map_no_agree = {0:0,
                    1:1,
                    2:2,
                    3:3,
                    4:3,
                    5:3}

state_map_no_init_or_agree = {
    0:0,
    1:0,
    2:0,
    3:1,
    4:2,
    5:3,
    6:3,
    7:3,
    8:2,
    9:3,
    10:3,
    11:3
}

# matrix of legal transitions
legal_transitions = np.zeros((12,6,12))
legal_transitions[:,0,0] = 1
legal_transitions[:,0,1] = 1
legal_transitions[:,0,2] = 1
legal_transitions[:,1,3] = 1
legal_transitions[:,2,4] = 1
legal_transitions[:,2,8] = 1
legal_transitions[:,3,5] = 1
legal_transitions[:,3,9] = 1
legal_transitions[:,4,6] = 1
legal_transitions[:,4,10] = 1
legal_transitions[:,5,7] = 1
legal_transitions[:,5,11] = 1

# matrix of legal transitions for reduced state configs
legal_trans_no_agree = np.zeros((6,4,6))
legal_trans_no_agree[:,0,0] = 1
legal_trans_no_agree[:,1,1] = 1
legal_trans_no_agree[:,2,2] = 1
legal_trans_no_agree[:,2,4] = 1
legal_trans_no_agree[:,3,3] = 1
legal_trans_no_agree[:,3,5] = 1

legal_trans_no_init = np.zeros((8,6,8))
legal_trans_no_init[:,0,0] = 1
legal_trans_no_init[:,0,1] = 1
legal_trans_no_init[:,0,2] = 1
legal_trans_no_init[:,1,3] = 1
legal_trans_no_init[:,2,4] = 1
legal_trans_no_init[:,3,5] = 1
legal_trans_no_init[:,4,6] = 1
legal_trans_no_init[:,5,7] = 1

legal_trans_no_init_or_agree = np.zeros((4,4,4))


