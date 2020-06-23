import numpy as np
import pandas as pd

def data_prep():
    return 1
    
def main():
    data_path = './'
    filename = 'task_data.csv'
    data = pd.read_csv(data_path+filename, index_col='sample index')
    print(data.head())
if __name__ == "__main__":
    main()
