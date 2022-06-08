#!/bin/bash

# input is protein (needs to be converted to pocket)
python rtmscore.py -p ./1qkt_p.pdb -l ./1qkt_decoys.sdf -rl ./1qkt_l.sdf -gen_pocket -c 10.0 -m ../trained_models/rtmscore_model1.pth

# input is pocket
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -m ../trained_models/rtmscore_model1.pth


# calculate the atom contributions of the score
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -ac -m ../trained_models/rtmscore_model1.pth


# calculate the residue contributions of the score
python rtmscore.py -p ./1qkt_p_pocket_10.0.pdb -l ./1qkt_decoys.sdf -rc -m ../trained_models/rtmscore_model1.pth

