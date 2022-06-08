import os,glob
import subprocess
from joblib import Parallel, delayed

def convert(pdbid):
	#mol2files = glob.glob("./%s/%s_*.mol2"%(pdbid, pdbid))
	ids2 = [x for x in os.listdir('./%s'%pdbid) if x.endswith(".mol2")]
	for i in ids2:
		cmd = "module load openeye &&"
		cmd += "convert.py %s %s"%(i, i.replace(".mol2",".sdf"))
		#os.system(cmd)
		p = subprocess.Popen([cmd], shell=True, cwd=pdbid)
		p.wait()

pdbids = [x for x in os.listdir('.') if os.path.isdir("./%s"%x)]		
#for pdbid in pdbids:
#	convert(pdbid)
Parallel(n_jobs=-1, backend="multiprocessing")(delayed(convert)(pdbid) for pdbid in pdbids)
