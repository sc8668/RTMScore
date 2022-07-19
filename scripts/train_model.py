import torch as th
import numpy as np
import dgl
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys
sys.path.append("/home/shenchao/resdocktest2/rtmscore2")
from RTMScore.data.data import PDBbindDataset
from RTMScore.model.model2 import RTMScore, DGLGraphTransformer 
from RTMScore.model.utils import collate, EarlyStopping, set_random_seed, run_a_train_epoch, run_an_eval_epoch, mdn_loss_fn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
args={}
args["num_epochs"] = 5000
args["batch_size"] = 64#128
args["aux_weight"] = 0.001
args['patience'] = 70 
args["num_workers"] = 10
args["model_path"] = "xxx.pth"
args['mode'] = "lower"
args['lr'] = 3
args['weight_decay'] = 5
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args['seeds'] = 126
args["data_dir"] = "/home/shenchao/resdocktest2/dataset"
args["train_prefix"] = "v2020_train"
#args["test1_prefix"] = "v2020_casf"
#args["test2_prefix"] = "v2020_core"
args["valnum"] = 1500
args["hidden_dim0"] = 128
args["hidden_dim"] = 128 
args["n_gaussians"] = 10
args["dropout_rate"] = 0.10
args["dist_threhold"] = 7.



data = PDBbindDataset(ids="%s/%s_ids.npy"%(args["data_dir"], args["train_prefix"]),
					  ligs="%s/%s_l.bin"%(args["data_dir"], args["train_prefix"]),
					  prots="%s/%s_p.bin"%(args["data_dir"], args["train_prefix"])
					  )

train_inds, val_inds = data.train_and_test_split(valnum=args["valnum"], seed=args['seeds'])
train_data = PDBbindDataset(ids=data.pdbids[train_inds],
							ligs=np.array(data.graphsl)[train_inds],
							prots=np.array(data.graphsp)[train_inds]							
							)
val_data = PDBbindDataset(ids=data.pdbids[val_inds],
							ligs=np.array(data.graphsl)[val_inds],
							prots=np.array(data.graphsp)[val_inds]
							)
							
#test_data1 = PDBbindDataset(ids="%s/%s_idsresx.npy"%(args["data_dir"], args["test1_prefix"]),
#					  graphs="%s/%sresx.bin"%(args["data_dir"], args["test1_prefix"])
#					  )
#test_data2 = PDBbindDataset(ids="%s/%s_idsresx.npy"%(args["data_dir"], args["test2_prefix"]),
#					  graphs="%s/%sresx.bin"%(args["data_dir"], args["test2_prefix"])
#					  )


ligmodel = DGLGraphTransformer(in_channels=41, 
								edge_features=10, 
								num_hidden_channels=args["hidden_dim0"],
								activ_fn=th.nn.SiLU(),
								transformer_residual=True,
								num_attention_heads=4,
								norm_to_apply='batch',
								dropout_rate=0.15,
								num_layers=6
								)

protmodel = DGLGraphTransformer(in_channels=41, 
									edge_features=5, 
									num_hidden_channels=args["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)

model = RTMScore(ligmodel, protmodel, 
				in_channels=args["hidden_dim0"], 
				hidden_dim=args["hidden_dim"], 
				n_gaussians=args["n_gaussians"], 
				dropout_rate=args["dropout_rate"],
				dist_threhold=args["dist_threhold"]).to(args['device'])

optimizer = th.optim.Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])



train_loader = DataLoader(dataset=train_data, 
							batch_size=args["batch_size"],
							shuffle=True, 
							num_workers=args["num_workers"],
							collate_fn=collate)

val_loader = DataLoader(dataset=val_data, 
							batch_size=args["batch_size"],
							shuffle=False, 
							num_workers=args["num_workers"],
							collate_fn=collate)


stopper = EarlyStopping(patience=args['patience'], mode=args['mode'], filename=args["model_path"])

set_random_seed(args["seeds"])
for epoch in range(args["num_epochs"]):	
	# Train
	total_loss_train, mdn_loss_train, atom_loss_train, bond_loss_train = run_a_train_epoch(epoch, model, train_loader, optimizer, aux_weight=args["aux_weight"], device=args["device"])	
	if np.isinf(mdn_loss_train) or np.isnan(mdn_loss_train): 
		print('Inf ERROR')
		break
	# Validation and early stop
	total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val = run_an_eval_epoch(model, val_loader, dist_threhold=args["dist_threhold"], aux_weight=args["aux_weight"], device=args["device"])
	early_stop = stopper.step(total_loss_val, model)
	print('epoch {:d}/{:d}, total_loss_val {:.4f}, mdn_loss_val {:.4f}, atom_loss_val {:.4f}, bond_loss_val {:.4f}, best validation {:.4f}'.format(epoch + 1, args['num_epochs'], total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val, stopper.best_score)) #+' validation result:', validation_result)
	if early_stop:
		break

		
stopper.load_checkpoint(model)
total_loss_train, mdn_loss_train, atom_loss_train, bond_loss_train = run_an_eval_epoch(model, train_loader, dist_threhold=args["dist_threhold"], aux_weight=args["aux_weight"], device=args["device"])
total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val = run_an_eval_epoch(model, val_loader, dist_threhold=args["dist_threhold"], aux_weight=args["aux_weight"], device=args["device"])		

print("total_loss_train:%s, mdn_loss_train:%s, atom_loss_train:%s, bond_loss_train:%s"%(total_loss_train, mdn_loss_train, atom_loss_train, bond_loss_train))
print("total_loss_val:%s, mdn_loss_val:%s, atom_loss_val:%s, bond_loss_val:%s"%(total_loss_val, mdn_loss_val, atom_loss_val, bond_loss_val))








