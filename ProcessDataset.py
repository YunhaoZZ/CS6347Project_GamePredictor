import pandas

def preprocess(path):
  df = pandas.read_csv(path)
  return df

path="all_seasons.csv"
df=preprocess(path)
df