import numpy as np
import sys, os
import pandas as pd
import argparse
import multiprocessing
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

## python ranking_power2.py -c ./CoreSet.dat -s ./examples/newdeepdock1.dat -p positive -o newdeepdock1 


def usage():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--coreset', default='/home/shenchao/test/CASF-2016/power_ranking/CoreSet.dat',
						help="specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package")
	parser.add_argument('-s', '--scorefile', default='/home/shenchao/test/CASF-2016/power_ranking/examples/X-Score.dat',
						help="specify the directory containing your scoring files(e.g. 'XXXX_score.dat').\
						Remember the 1st column name is #code and the 2nd column name is score.\
						 Supported file separators are comma(,), tabs(\\t) and space character( )")
	parser.add_argument('-p', '--prefer', default='positive', choices=['positive','negative'],
						help="input 'negative' or 'positive' string, depend on your scoring funtion preference")										
	parser.add_argument('-o', '--output', default='X-Score',
						help="input the prefix of output result files. Default is My_Docking_Power")
	
	#parser.add_argument('-i', '--i', default=10000, type=int,
	#								help='The reample times.')
	args = parser.parse_args()
	return args			


def cal_PI(score, logKa):
	"""Define the Predictive Index function"""
	logKa, score = zip(*sorted(zip(logKa,score), key=lambda x:x[0], reverse=False))
	W=[]
	WC=[]
	for i in np.arange(0,5):
		for j in np.arange(i+1,5):
			w_ij=abs(logKa[i]-logKa[j])
			W.append(w_ij)
			if score[i] < score[j]:
				WC.append(w_ij)
			elif score[i] > score[j]:
				WC.append(-w_ij)
			else:
				WC.append(0)
	
	pi=float(sum(WC))/float(sum(W))
	return pi	
				

def main():
	args = usage()
	#i_list = [int(x) for x in np.linspace(0,100000, args.i)]	
	df = pd.read_csv(args.coreset, sep='[,,\t, ]+', header=0, engine='python')
	df_score = pd.read_csv(args.scorefile,sep='[,, ,\t]+',engine='python')
	
	testdf = pd.merge(df,df_score,on='#code')

	df_groupby = testdf.groupby('target')
	
	spearman = df_groupby.apply(lambda x: x[["logKa","score"]].corr("spearman").iloc[1,0])
	kendall = df_groupby.apply(lambda x: x[["logKa","score"]].corr("kendall").iloc[1,0])
	PI = df_groupby.apply(lambda x: cal_PI(x.score, x.logKa))
	
	print('casf2016_ranking_%s'%args.output)
	print("The Spearman correlation coefficient (SP): %.3f"%spearman.mean())
	print("The Kendall correlation coefficient (tau): %.3f"%kendall.mean())
	print("The Predictive index (PI): %.3f"%PI.mean())


		
if __name__ == '__main__':
    main()


