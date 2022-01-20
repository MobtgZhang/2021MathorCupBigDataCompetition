from asyncore import write
import os
from statistics import mode
import pandas as pd
def main():
    filename= os.path.join("result","predict-v1.3.csv")
    dataset = pd.read_csv(filename)
    with open("AMCB2101969Test.txt",mode="w",encoding="utf-8") as wfp:
        for k in range(len(dataset)):
            wfp.write(str(dataset.iloc[k,0])+"\t"+str(dataset.iloc[k,1])+"\n")
if __name__ == "__main__":
    main()
