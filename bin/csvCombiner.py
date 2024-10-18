import pandas as pd 
import os
import sys 

def csvCombiner (outname):
    files = os.listdir()
    csvFiles = [file for file in files if ".csv" in file]
    df = pd.read_csv(csvFiles[0])
    for i in range (1,len(csvFiles)): 
        df_sub = pd.read_csv(csvFiles[i])
        df = pd.concat((df,df_sub),axis=0)
    df.to_csv(outname,index=False)
    
csvCombiner(sys.argv[1])