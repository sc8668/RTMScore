#!/usr/bin/python
import numpy as np
import sys, os
import pandas as pd
import argparse
#import multiprocessing

## python forward_screening_power2.py -c ./CoreSet.dat -t ./TargetInfo.dat -s ./examples/newdeepdock1 -p positive -o newdeepdock1 


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
	parser.add_argument('-t', '--targetfile', default='/home/shenchao/test/CASF-2016/power_screening/TargetInfo.dat',
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
		success_mol = df_groupby.apply(lambda x: 1 if x.topactid.values[0] in x.nsmallest(round(top * len(x)),score_name).index else 0).sum()
	else:
		success_mol = df_groupby.apply(lambda x: 1 if x.topactid.values[0] in x.nlargest(round(top * len(x)),score_name).index else 0).sum()
	
	return success_mol, success_mol/total_mol


def calc_ef(df_total, score_name, label_name, pos_label="positive", threshold=0.01):
    '''
    calculate the enrichment factor
    '''
    N_total = len(df_total)
    N_actives = len(df_total[df_total[label_name] == 1])
    if pos_label == 'negative':
        total_sorted = df_total.sort_values(by=[score_name], ascending=True)
    else:
        total_sorted = df_total.sort_values(by=[score_name], ascending=False)
	
    #N_topx_total = int(np.ceil(N_total * threshold))
    N_topx_total = round(N_total * threshold)
    topx_total = total_sorted.iloc[:N_topx_total, :]
    N_topx_actives = len(topx_total[topx_total[label_name] == 1])
	
    return N_topx_actives / (N_actives * threshold)
    #return (N_topx_actives / N_topx_total) / (N_actives / N_total)


def obtain_metircs(args, df):	
	#df = resample(df, random_state=i, replace=True)
	df_groupby = df.groupby('target')
	topnum1, SR1 = calc_success_rate(df_groupby, 'score', args.prefer, top=0.01)
	topnum5, SR5 = calc_success_rate(df_groupby, 'score', args.prefer, top=0.05)
	topnum10, SR10 = calc_success_rate(df_groupby, 'score', args.prefer, top=0.1)	
	
	EF1 = df_groupby.apply(lambda x: calc_ef(x, 'score', "label", args.prefer, threshold=0.01)).mean()
	EF5 = df_groupby.apply(lambda x: calc_ef(x, 'score', "label", args.prefer, threshold=0.05)).mean()
	EF10 = df_groupby.apply(lambda x: calc_ef(x, 'score', "label", args.prefer, threshold=0.1)).mean()
	return topnum1, SR1*100, topnum5, SR5*100, topnum10, SR10*100, EF1, EF5, EF10
	#return_dict[i] = [Top1, Top2, Top3] + list(sp_dict.values())
	


def main():
	args = usage()
	#i_list = [int(x) for x in np.linspace(0,100000, args.i)]		
	df = pd.read_csv(args.coreset, sep='[,,\t, ]+', header=0, engine='python')
	topligs = df.groupby("target").apply(lambda x: x.sort_values("logKa").iloc[-1])
	dft = pd.read_csv(args.targetfile, sep='[,,\t, ]+', header=0, skiprows=8, engine='python')
	dft = dft.drop_duplicates(subset=['#T'],keep='first')	
	dft["target"] = [int(df[df["#code"]==i].target) for i in dft['#T']]	
	df_list = []
	for i in dft['#T']:
		act_id = dft[dft["#T"]==i].iloc[:,:-1].dropna(axis=1).values.reshape(-1)
		#topact_id = topligs[topligs.target == int(dft[dft["#T"]==i].loc[:,"target"])]["#code"].values[0]
		topact_id = dft[dft["#T"]==i].loc[:,"L1"].values[0]
		#dec_id = np.setdiff1d(df["#code"], act_id)
		df_score = pd.read_csv(args.scoredir+'/'+str(i)+'_score.dat',sep='[,, ,\t]+',engine='python')
		df_score['ligid'] = df_score["#code_ligand_num"].apply(lambda x: x.split("_")[0])
		if args.prefer == 'negative':
			df_score = df_score.groupby("ligid").min()
		else:
			df_score = df_score.groupby("ligid").max()

		df_score['topactid'] = topact_id
		df_score['target'] = int(dft[dft["#T"]==i].loc[:,"target"])
		df_score['label'] = df_score["#code_ligand_num"].apply(lambda x: 1 if x.split("_")[0].strip() in act_id else 0)	
		df_list.append(df_score)

	dfx = pd.concat(df_list, axis=0)
	topnum1, SR1, topnum5, SR5, topnum10, SR10, EF1, EF5, EF10 = obtain_metircs(args, dfx)
	
	print('casf2016_screening_forward_%s'%args.output)
	print("Average enrichment factor among top 1%%: %.2f"%EF1)
	print("Average enrichment factor among top 5%%: %.2f"%EF5)
	print("Average enrichment factor among top 10%%: %.2f"%EF10)
	print("The best ligand is found among top 1%% candidates for %2d cluster(s)"%topnum1)
	print("		top 1%% sucess rate: %.1f%%"%SR1)
	print("The best ligand is found among top 5%% candidates for %2d cluster(s)"%topnum5)
	print("		top 5%% sucess rate: %.1f%%"%SR5)	
	print("The best ligand is found among top 10%% candidates for %2d cluster(s)"%topnum10)
	print("		top 10%% sucess rate: %.1f%%"%SR10)



	
if __name__ == '__main__':
    main()


