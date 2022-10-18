from unicodedata import numeric
import pandas as pd
import numpy as np
from tqdm import tqdm
import re

df = pd.read_csv('data/small_data.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

numeric_cols = list(df.columns)[:-1]
funcs = {'mean':np.mean, 'std':np.std, 'max':np.max, 'min':np.min}

df['time_group'] = df.index.to_period('H').strftime('%d_%H')

summary = {key: [] for key in funcs.keys()}
summary['info'] = []
summary['target'] = []

for time_slice in tqdm(df['time_group'].unique()):
    for device in (df['device_mac'].unique()):
        tmp = df.loc[(df['time_group']==time_slice)&(df['device_mac']==device)][numeric_cols]
        for col in tmp.columns:
            for k in funcs.keys():
                summary[k].append(funcs[k](tmp[col]))
            summary['info'].append(device+'_'+time_slice)
            summary['target'].append(col)

summary = pd.DataFrame(summary)
summary.to_csv('data/transformed_data.csv',index=False)

print(summary)

