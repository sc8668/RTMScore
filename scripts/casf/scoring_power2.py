import numpy as np
import sys, os
import pandas as pd
import argparse
import multiprocessing
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
#from multiprocessing import Manager
#from sklearn.utils import resample

## python scoring_power2.py -c ./CoreSet.dat -s ./examples/newdeepdock1.dat -p positive -o newdeepdock1 -remove_outliners

def usage():
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--coreset', default='/home/shenchao/test/CASF-2016/power_scoring/CoreSet.dat',
						help="specify the location of 'CoreSet.dat' (or a subset data file) in the CASF-2016 package")
	parser.add_argument('-s', '--scorefile', default='/home/shenchao/test/CASF-2016/power_scoring/examples/X-Score.dat',
						help="specify the directory containing your scoring files(e.g. 'XXXX_score.dat').\
						Remember the 1st column name is #code and the 2nd column name is score.\
						 Supported file separators are comma(,), tabs(\\t) and space character( )")
	parser.add_argument('-p', '--prefer', default='positive', choices=['positive','negative'],
						help="input 'negative' or 'positive' string, depend on your scoring funtion preference")										
	parser.add_argument('-o', '--output', default='X-Score',
						help="input the prefix of output result files. Default is My_Docking_Power")													
	parser.add_argument('-remove_outliners', '--remove_outliners', action='store_true', default=False, 
						help="whether to remove the outliners (in the original CASF the outliners are removed)")	
	
	#parser.add_argument('-i', '--i', default=10000, type=int,
	#								help='The reample times.')
	args = parser.parse_args()
	return args					


def obtain_metrics(df):
	#Calculate the Pearson correlation coefficient
	regr = linear_model.LinearRegression()
	regr.fit(df.score.values.reshape(-1,1), df.logKa.values.reshape(-1,1))
	preds = regr.predict(df.score.values.reshape(-1,1))
	rp = pearsonr(df.logKa, df.score)[0]
	#rp = df[["logKa","score"]].corr().iloc[0,1]
	mse = mean_squared_error(df.logKa, preds)
	num = df.shape[0]
	sd = np.sqrt((mse*num)/(num-1))
	#return rp, sd, num
	print("The regression equation: logKa = %.2f + %.2f * Score"%(float(regr.coef_), float(regr.intercept_)))
	print("Number of favorable sample (N): %d"%num)
	print("Pearson correlation coefficient (R): %.3f"%rp)
	print("Standard deviation in fitting (SD): %.2f"%sd)
	

def main():
	args = usage()
	#i_list = [int(x) for x in np.linspace(0,100000, args.i)]	
	df = pd.read_csv(args.coreset, sep='[,,\t, ]+', header=0, engine='python')
	df_score = pd.read_csv(args.scorefile,sep='[,, ,\t]+',engine='python')
	
	testdf = pd.merge(df,df_score,on='#code')
	if args.prefer == "negative":
		testdf.score = -testdf.score
	if args.remove_outliners:
		testdf = testdf[testdf.score > 0]
	
	print('casf2016_scoring_%s'%args.output)
	obtain_metrics(testdf)

		
if __name__ == '__main__':
    main()
