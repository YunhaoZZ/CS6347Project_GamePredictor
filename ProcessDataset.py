import pandas

def preprocess(path):
  fields = ['App ID', 'Tags', 'Positive Reviews','Negative Reviews']
  df = pandas.read_csv(path, usecols=fields, delimiter=';')
  return df

path="steam_games.csv"
df=preprocess(path)
df = df.dropna()
# print(df['Tags'].head())

# record tags into a dictionary
tags = {}
for row in df['Tags']:
  tag_pairs = row.split(',')
  for tag_pair in tag_pairs:
    tag_items = tag_pair.split(':')
    if tag_items[0] not in tags.keys():
      tags[tag_items[0]] = 1
    else:
      tags[tag_items[0]] += 1


print(tags.keys())
print(max(tags.values()))
print(min(tags.values()))