import pandas

def preprocess(path):
  df = pandas.read_csv(path)
  return df

path="steam_games.csv"
df=preprocess(path)
df