# this script detects which line caused HuggingFace Transformers to crash
import pandas as pd

# go through the file and find rows where entries are not either strings or integers
FILE_PATH = "../datasets/imputed/training/SChem_training_imputed_train.csv"

# open the file
df = pd.read_csv(FILE_PATH)

# make sure that the only columns are "sentence1", "sentence2" (optional), and "label"
if False in [col in ["sentence1", "sentence2", "label"] for col in df.columns]:
    print(f"The file should only have columns named \"sentence1\", \"sentence2\", and \"label\". Instead, the columns are: {df.columns}")

# find the rows where the column labeled "sentence1" is not a string
df_bad_lines = df[df["sentence1"].apply(lambda x: type(x) != str)]

if "sentence2" in df.columns:
    # add in the rows where the column labeled "sentence2" is not a string
    df_bad_lines = pd.concat([df_bad_lines, df[df["sentence2"].apply(lambda x: type(x) != str)]])

# add in the rows where the column labeled "label" is not a float
# use concat instead of append because append is deprecated
df_bad_lines = pd.concat([df_bad_lines, df[df["label"].apply(lambda x: type(x) != float)]])


# print all the bad lines as a list
print(f"All bad lines:\n{df_bad_lines.index.tolist()}")

# display the line number and the contents of each bad row
for index, row in df_bad_lines.iterrows():
    print(index)
    print(row)