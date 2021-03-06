import pandas as pd
import os
import numpy as np
from dpdi.datasets.utils import normalize_columns

def load_law_dataset(sensitive, root_dir="../data/ucla-law-school", normalize=True):
    """Reads the law school dataset, as in (Khani et al. 2020).

    Adapted from
    https://worksheets.codalab.org/rest/bundles/0xff62528c13b84510b7f10562f21be280
    /contents/blob/data_preprocess/student.py
    """
    data = pd.read_csv(os.path.join(root_dir, 'lsac.csv'), na_values=" ")
    data.dropna(axis=0, inplace=True)
    data = data[[
        'sex', 'race', 'cluster',
        'lsat', 'ugpa', 'zfygpa', 'dob_yr', 'zgpa', 'bar1', 'bar1_yr',
        'bar2', 'bar2_yr', 'fam_inc', 'parttime',
        'pass_bar', 'bar', 'tier']]
    data = pd.get_dummies(data)
    if sensitive == 'sex':
        data['sex'] -= 1
        data.rename(columns={'sex': 'sensitive'}, inplace=True)
    elif sensitive == 'race':
        print("[INFO] with sens. attr. == race, '1' is non-white, '0' is white.")
        data.loc[data.race < 7, 'race'] = 1  # non-white
        data.loc[data.race >= 7, 'race'] = 0  # white
        data.rename(columns={'race': 'sensitive'}, inplace=True)

    data.rename(columns={'zgpa': 'target'}, inplace=True)
    if normalize:
        data = normalize_columns(data)
    return data
