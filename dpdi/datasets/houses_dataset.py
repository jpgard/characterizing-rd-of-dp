import pandas as pd
import os

def load_houses_dataset(root_dir, census_csv):
    houses = pd.read_csv(os.path.join(root_dir, "kc-houses.csv"))
    census = pd.read_csv(census_csv)
