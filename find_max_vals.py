import csv
import pandas as pd
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file_path",
    default='metrics.csv',
	help="the relative path of metrics.csv")
parser.add_argument('-c','--cols', nargs='+',
    default=['val_acc'],
    help='column names that you want to find max value from')
args = vars(parser.parse_args())

print("\nopening {} ...\n".format(args["file_path"]))
print("checking {} ...\n".format(args["cols"]))


which_file = args["file_path"]
cols_want = args["cols"]
df = pd.read_csv(which_file)
df.fillna(0, inplace=True)
data = df.values
cols = df.columns.values
for col_name in cols_want:
    which_col = np.where(cols==col_name)[0][0]
    that_col = data[:, which_col]
    max_val = np.max(that_col)
    which_row = np.argmax(that_col, axis=0)
    that_row = data[which_row,:]
    where_epoch = np.where(cols=="epoch")[0][0]
    epoch_col = data[:, where_epoch]
    min_epoch_val = np.min(epoch_col)
    curr_epoch_val = that_row[where_epoch]
    print("max value for {} is {} in epoch {},"\
        .format(col_name, max_val, int(curr_epoch_val)))
    print("epoch number starts from {} in {}".\
        format(str(int(min_epoch_val)), which_file))
    print()