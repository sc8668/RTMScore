#!/usr/bin/python
import numpy as np
import sys, os
import pandas as pd
import argparse
import multiprocessing
#from multiprocessing import Manager
#from sklearn.utils import resample

## python docking_power2.py -c ./CoreSet.dat -s ./examples/newdeepdock1 -p positive -r ../decoys_docking -o newdeepdock1 -l 2.0 

def usage():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--coreset', default='/home/shenchao/AI_pose_filter/CASF-2016/power_docking/CoreSet.dat',
						help="specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package")
	parser.add_argument('-s', '--scoredir', default='/home/shenchao/AI_pose_filter/CASF-2016/power_docking/examples/X-Score',
						help="specify the directory containing your scoring files(e.g. 'XXXX_score.dat').\
						Remember the 1st column name is #code and the 2nd column name is score.\
						 Supported file separators are comma(,), tabs(\\t) and space character( )")
	parser.add_argument('-p', '--prefer', default='positive', choices=['positive','negative'],
						help="input 'negative' or 'positive' string, depend on your scoring funtion preference")						
	parser.add_argument('-r', '--rmsddir', default='/home/shenchao/AI_pose_filter/CASF-2016/decoys_docking',
						help="specify the directory containing the RMSD data files(e.g. 'XXXX_rmsd.dat')")						
	parser.add_argument('-o', '--output', default='X-Score',
						help="input the prefix of output result files. Default is My_Docking_Power")													
	parser.add_argument('-l', '--limit', default=2.0, type=float,
						help="set the RMSD cutoff (in angstrom) to define near-native docking pose")
	parser.add_argument('-remain_crystalposes', '--remain_crystalposes', action='store_true', default=False,
									help='whether to remain the crystalized poses.') 
	#parser.add_argument('-i', '--i', default=10000, type=int,
	#								help='The reample times.')
	args = parser.parse_args()
	return args					


def calc_success_rate2(df_groupby, score_name, pos_label, rmsd_cut=2.0, topn=1):
	'''只要挑选出来的topn构象有满足RMSD<=2的既可.'''
	total_mol = len(df_groupby)
	if pos_label == 'negative':
		success_mol = df_groupby.apply(lambda x: 1 if x.rmsd.loc[x.nsmallest(topn,score_name).index].min() <= rmsd_cut else 0).sum()
	else:
		success_mol = df_groupby.apply(lambda x: 1 if x.rmsd.loc[x.nlargest(topn,score_name).index].min() <= rmsd_cut else 0).sum()
	
	return success_mol/total_mol


def calc_sp(df, pos_label, cutoff=2.0):
	df0 = df[df.rmsd<=cutoff]
	if float(df0.shape[0]) >= 5:
		if pos_label == 'negative':
			return df0.corr('spearman').iloc[0,1]
		else:
			return -df0.corr('spearman').iloc[0,1]


#def obtain_metircs(args, df, i, return_dict):	
def obtain_metircs(args, df):	
	#df = resample(df, random_state=i, replace=True)
	df_groupby = df.groupby('pdb')
	Top1=calc_success_rate2(df_groupby, 'score', args.prefer, args.limit, topn=1)	
	Top2=calc_success_rate2(df_groupby, 'score', args.prefer, args.limit, topn=2)	
	Top3=calc_success_rate2(df_groupby, 'score', args.prefer, args.limit, topn=3)	
	sp_dict = {}
	for s in np.arange(2,11):
		sp = df_groupby.apply(lambda x: calc_sp(x, args.prefer, cutoff=s)).mean()
		sp_dict[s] = sp
	return [Top1, Top2, Top3] + list(sp_dict.values())
	#return_dict[i] = [Top1, Top2, Top3] + list(sp_dict.values())
	


def main():
	args = usage()
	#i_list = [int(x) for x in np.linspace(0,100000, args.i)]	
	df = pd.read_csv(args.coreset, sep='[,,\t, ]+', header=0, engine='python')
	df_list = []
	for i in df['#code']:
		df_rmsd = pd.read_csv(args.rmsddir+'/'+str(i)+'_rmsd.dat',sep='[,, ,\t]+', header=0, engine='python')
		df_score = pd.read_csv(args.scoredir+'/'+str(i)+'_score.dat',sep='[,, ,\t]+',engine='python')
		df0 = pd.merge(df_rmsd,df_score,on='#code')
		df_list.append(df0)
	
	dfx = pd.concat(df_list,axis=0)
	dfx['pdb'] = dfx['#code'].apply(lambda x:x.split('_')[0])
	dfx.sort_values('#code',inplace=True)
	if not args.remain_crystalposes:
		dfx = dfx[~dfx['#code'].str.contains("_ligand")]
	
	#manger = Manager()
	#return_dict = manger.dict()
	#pool = multiprocessing.Pool(32)
	#jobs = []
	#for i in i_list:
	#	p = pool.apply_async(obtain_metircs, (args, dfx, i, return_dict))
	#	jobs.append(p)
	#	#p.start()
	#	
	#pool.close()
	#pool.join()	
	#
	#results = return_dict.values()
	results = obtain_metircs(args, dfx)
	#final_values = list(zip(*results))
	
	print('casf2016_docking_%s'%args.output)
	print("Top1 success rate: %.3f"%results[0])
	print("Top2 success rate: %.3f"%results[1])
	print("Top3 success rate: %.3f"%results[2])
	print("Spearman correlation coefficient in rmsd range [0-2]: %.3f"%results[3])
	print("Spearman correlation coefficient in rmsd range [0-3]: %.3f"%results[4])
	print("Spearman correlation coefficient in rmsd range [0-4]: %.3f"%results[5])
	print("Spearman correlation coefficient in rmsd range [0-5]: %.3f"%results[6])
	print("Spearman correlation coefficient in rmsd range [0-6]: %.3f"%results[7])
	print("Spearman correlation coefficient in rmsd range [0-7]: %.3f"%results[8])
	print("Spearman correlation coefficient in rmsd range [0-8]: %.3f"%results[9])
	print("Spearman correlation coefficient in rmsd range [0-9]: %.3f"%results[10])
	print("Spearman correlation coefficient in rmsd range [0-10]: %.3f"%results[11])
 
	#out_df = pd.DataFrame(results)
	#out_df.columns = ['sr_top1','sr_top2','sr_top3','Rs_0_2','Rs_0_3','Rs_0_4','Rs_0_5','Rs_0_6','Rs_0_7','Rs_0_8','Rs_0_9','Rs_0_10']
	###out_df.loc['mean'] = out_df.apply(lambda x: x.mean())
	####out_df.loc['std'] = out_df.apply(lambda x: x.std())
	#out_df.to_csv(args.output)	


	
	
if __name__ == '__main__':
    main()
