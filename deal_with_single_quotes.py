# quick script that deals with single quotes that may mess up Hugging Face Transformers

import pandas as pd

# set these to your dataset and where you'd like the fixed dataset saved
YOUR_CSV_DATASET = "your_csv_dataset.csv"
FIXED_CSV_DATASET = "your_csv_dataset_fixed.csv"

# constants for saving the dataset correctly - do not modify
QUOTE_CHAR = '"'
# from https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
QUOTE_ALL = 1

# save the dataset in a way that Hugging Face Transformers can read (i.e. with double quotes)
df = pd.read_csv(YOUR_CSV_DATASET)
df.to_csv(FIXED_CSV_DATASET, quoting=QUOTE_ALL, quotechar=QUOTE_CHAR)