import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
sys.path.append("/home/shenchao/resdocktest2/rtmscore2")
from torch.utils.data import DataLoader
from RTMScore.data.data import VSDataset
from RTMScore.model.utils import collate, run_an_eval_epoch
from RTMScore.model.model2 import RTMScore, DGLGraphTransformer #LigandNet, TargetNet
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
args={}
args["batch_size"] = 128
args["aux_weight"] = 0.001
args["dist_threhold"] = 5
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args['seeds'] = 126
args["num_workers"] = 10
args["model_path"] = "/home/shenchao/resdocktest/deepdock2/trained_models/rtmscore_model1.pth"
args["cutoff"] = 10.0
args["num_node_featsp"] = 41
args["num_node_featsl"] = 41
args["num_edge_featsp"] = 5
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128 
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.10
args["outprefix"] = "rtmscore1x5"


def scoring(prot, lig, modpath,
			cut=10.0,
			explicit_H=False, 
			use_chirality=True,
			parallel=False,
			**kwargs
			):
	"""
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	reflig: The reference ligand to determine the pocket.
	cutoff: The distance within the reference ligand to determine the pocket.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.	
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""
	#try:
	data = VSDataset(ligs=lig,
					prot=prot,
					cutoff=cut,		
					explicit_H=explicit_H, 
					use_chirality=use_chirality,
					parallel=parallel)
						
	test_loader = DataLoader(dataset=data, 
							batch_size=kwargs["batch_size"],
							shuffle=False, 
							num_workers=kwargs["num_workers"],
							collate_fn=collate)
	
	ligmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsl"], 
									edge_features=kwargs["num_edge_featsl"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
	
	protmodel = DGLGraphTransformer(in_channels=kwargs["num_node_featsp"], 
									edge_features=kwargs["num_edge_featsp"], 
									num_hidden_channels=kwargs["hidden_dim0"],
									activ_fn=th.nn.SiLU(),
									transformer_residual=True,
									num_attention_heads=4,
									norm_to_apply='batch',
									dropout_rate=0.15,
									num_layers=6
									)
						
	model = RTMScore(ligmodel, protmodel, 
					in_channels=kwargs["hidden_dim0"], 
					hidden_dim=kwargs["hidden_dim"], 
					n_gaussians=kwargs["n_gaussians"], 
					dropout_rate=kwargs["dropout_rate"], 
					dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])
	
	checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
	model.load_state_dict(checkpoint['model_state_dict']) 
	preds = run_an_eval_epoch(model, test_loader, pred=True, dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])	
	return data.ids, preds
	#except:
	#	print("failed to scoring for {} and {}".format(prot, lig))
	#	return None, None



def score_compound(pdbid, prefix):
	return scoring(prot="/home/shenchao/pdbbind/%s/%s/%s_prot/%s_p_pocket_10.0.pdb"%(prefix, pdbid, pdbid, pdbid), 
					lig="/home/shenchao/test/CASF-2016/decoys_docking/%s_decoys.sdf"%pdbid,
					modpath=args["model_path"],
					cut=args["cutoff"],
					explicit_H=False, 
					use_chirality=True,
					parallel=True,
					**args
					)

def score_compound0(pdbid, prefix):
	ids, scores = scoring(prot="/home/shenchao/pdbbind/%s/%s/%s_prot/%s_p_pocket_10.0.pdb"%(prefix, pdbid, pdbid, pdbid), 
					lig="/home/shenchao/pdbbind/%s/%s/%s_prot/%s_l.sdf"%(prefix, pdbid, pdbid, pdbid), 
					modpath=args["model_path"],
					cut=args["cutoff"],
					explicit_H=False, 
					use_chirality=True,
					parallel=False,
					**args
					)
	ids.pop(-1)
	ids.append("%s_ligand"%pdbid)
	return ids, scores


def score_compoundxxx(pdbid, prefix):
	ids1, scores1 = score_compound(pdbid, prefix)
	ids2, scores2 = score_compound0(pdbid, prefix)
	return ids1+ids2, np.append(scores1,scores2)



pdbids = [x for x in os.listdir("/home/shenchao/test/CASF-2016/coreset") if os.path.isdir("/home/shenchao/test/CASF-2016/coreset/%s"%(x))]	
pdbids1 = [x for x in os.listdir("/home/shenchao/pdbbind/v2020-refined") if os.path.isdir("/home/shenchao/pdbbind/v2020-refined/%s"%(x))]	
pdbids2 = [x for x in os.listdir("/home/shenchao/pdbbind/v2020-other-PL") if os.path.isdir("/home/shenchao/pdbbind/v2020-other-PL/%s"%(x))]	

ids1 = [pdbid for pdbid in pdbids if pdbid in pdbids1]
ids2 = [pdbid for pdbid in pdbids if pdbid in pdbids2]
if args['device'] == 'cpu':
	results1 = Parallel(n_jobs=-1)(delayed(score_compoundxxx)(pdbid, "v2020-refined") for pdbid in ids1)
	results2 = Parallel(n_jobs=-1)(delayed(score_compoundxxx)(pdbid, "v2020-other-PL") for pdbid in ids2)
	results = results1 + results2
else:
	results = []
	for pdbid in ids1:
		results.append(score_compoundxxx(pdbid, "v2020-refined"))
	for pdbid in ids2:
		results.append(score_compoundxxx(pdbid, "v2020-other-PL"))

outdir = "/home/shenchao/test/CASF-2016/power_docking/examples/%s"%args["outprefix"]
os.system("mkdir -p %s"%outdir)
for res in results:
	pdbid = res[0][0].split("_")[0]
	df = pd.DataFrame(zip(*res),columns=["#code","score"])
	df["#code"] = df["#code"].str.split("-").apply(lambda x : x[0])
	df.to_csv("%s/%s_score.dat"%(outdir, pdbid), index=False, sep="\t")

with open("%s_docking.pkl"%args["outprefix"],"wb") as dbFile:
	pickle.dump(results,dbFile)

