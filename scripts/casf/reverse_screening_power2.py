#!/usr/bin/python
import numpy as np
import sys, os
import pandas as pd
import argparse
#import multiprocessing

## python reverse_screening_power2.py -c ./CoreSet.dat -l ./LigandInfo.dat -s ./examples/newdeepdock1 -p positive -o newdeepdock1 

def usage():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--coreset', default='/home/shenchao/test/CASF-2016/power_screening/CoreSet.dat',
						help="specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package")
	parser.add_argument('-s', '--scoredir', default='/home/shenchao/test/CASF-2016/power_screening/examples/X-Score',
						help="specify the directory containing your scoring files(e.g. 'XXXX_score.dat').\
						Remember the 1st column name is #code and the 2nd column name is score.\
						 Supported file separators are comma(,), tabs(\\t) and space character( )")
	parser.add_argument('-p', '--prefer', default='positive', choices=['positive','negative'],
						help="input 'negative' or 'positive' string, depend on your scoring funtion preference")										
	parser.add_argument('-o', '--output', default='X-Score',
						help="input the prefix of output result files. Default is My_Docking_Power")													
	parser.add_argument('-l', '--ligandinfo', default='/home/shenchao/test/CASF-2016/power_screening/LigandInfo.dat',
						help="specify the location of 'TargetInfo.dat' in the CASF-2016 package")		
	#parser.add_argument('-i', '--i', default=10000, type=int,
	#								help='The reample times.')
	args = parser.parse_args()
	return args		


def calc_success_rate(df_groupby, score_name, pos_label="positive", top=0.01):
	'''只要挑选出来的topn分子中有活性分子既可.
	CASF里面的意思好像是必须要有那个自己的分子(L1)。
	'''
	total_mol = len(df_groupby)
	#topn = round(top * total_mol)
	if pos_label == 'negative':
		success_mol = df_groupby.apply(lambda x: 1 if x.topactid.values[0] in x.nsmallest(round(top * len(x)),score_name).target.values else 0).sum()
	else:
		success_mol = df_groupby.apply(lambda x: 1 if x.topactid.values[0] in x.nlargest(round(top * len(x)),score_name).target.values else 0).sum()
	
	return success_mol, success_mol/total_mol


def obtain_metircs(args, df):	
	#df = resample(df, random_state=i, replace=True)
	df_groupby = df.groupby('ligid')
	topnum1, SR1 = calc_success_rate(df_groupby, 'score', args.prefer, top=0.01)
	topnum5, SR5 = calc_success_rate(df_groupby, 'score', args.prefer, top=0.05)
	topnum10, SR10 = calc_success_rate(df_groupby, 'score', args.prefer, top=0.1)	
	
	return topnum1, SR1*100, topnum5, SR5*100, topnum10, SR10*100
	


def main():
	args = usage()
	#i_list = [int(x) for x in np.linspace(0,100000, args.i)]		
	df = pd.read_csv(args.coreset, sep='[,,\t, ]+', header=0, engine='python')
	dfl = pd.read_csv(args.ligandinfo, sep='[,,\t, ]+', header=0, skiprows=8, engine='python')
	dfl = dfl.drop_duplicates(subset=['#code'],keep='first')
	df_list = []
	for i in sorted(set(dfl['T1'])):
		df_score = pd.read_csv(args.scoredir+'/'+str(i)+'_score.dat',sep='[,, ,\t]+',engine='python')
		df_score['ligid'] = df_score["#code_ligand_num"].apply(lambda x: x.split("_")[0])
		if args.prefer == 'negative':
			df_score = df_score.groupby("ligid").min()
		else:
			df_score = df_score.groupby("ligid").max()

		#df_score['topactid'] = topact_id
		df_score['target'] = i
		df_list.append(df_score)
	
	dfx = pd.concat(df_list, axis=0)
	dfx.reset_index(inplace=True)
	dfx['topactid'] = dfx.ligid.apply(lambda x: dfl[dfl["#code"]==x]["T1"].values[0]) 
	topnum1, SR1, topnum5, SR5, topnum10, SR10 = obtain_metircs(args, dfx)

	print('casf2016_screening_reverse_%s'%args.output)
	print("The best ligand is found among top 1%% candidates for %2d cluster(s)"%topnum1)
	print("		top 1%% sucess rate: %.1f%%"%SR1)
	print("The best ligand is found among top 5%% candidates for %2d cluster(s)"%topnum5)
	print("		top 5%% sucess rate: %.1f%%"%SR5)	
	print("The best ligand is found among top 10%% candidates for %2d cluster(s)"%topnum10)
	print("		top 10%% sucess rate: %.1f%%"%SR10)



	
if __name__ == '__main__':
    main()



