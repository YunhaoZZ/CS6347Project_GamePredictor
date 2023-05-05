import pandas
import numpy as np
from collections import Counter

def preprocess(path):
  fields = ['App ID', 'Tags', 'Positive Reviews','Negative Reviews']
  df = pandas.read_csv(path, usecols=fields, delimiter=';')
  return df

def getDataset(): 
  path="steam_games.csv"
  df=preprocess(path)
  df = df.dropna()
  df = df.reset_index(drop=True)
  # print(df['Tags'].head())

  # record tags into a dictionary
  tags = {}
  gTag = []
  for row in df['Tags']:
    tag_pairs = row.split(',')
    temp = []
    for tag_pair in tag_pairs:
      tag_items = tag_pair.split(':')
      if tag_items[0] not in tags.keys():
        tags[tag_items[0]] = 1
      else:
        tags[tag_items[0]] += 1
      temp.append(tag_items[0])
    gTag.append(temp)

  tag_counter = Counter(tags)
  tags = tag_counter.most_common(500)

  tag_id = {}
  for idx, tag in enumerate(tags):
    tag_id[idx] = tag[0]

  df = df.drop(columns=['Tags'])
  df['Tags'] = gTag

  rm = []
  y = []
  for idx, row in enumerate(df['Tags']):
    # temp = np.zeros(500)
    temp = [0 for _ in range(500)]
    for tag in row:
      if tag in tag_id.values():
        temp[list(tag_id.keys())[list(tag_id.values()).index(tag)]] = 1
    y.append(temp)
    if 1 not in temp:
      rm.append(idx)

  # y = np.array(y)
  df['y'] = y

  df = df.drop(rm)
  df = df.reset_index(drop=True)

  rm = []
  x = []
  for idx, row in df.iterrows():
    pr = row['Positive Reviews']
    nr = row['Negative Reviews']
    total = pr + nr
    if total == 0:
      rm.append(idx)
      total = 1
    rating = (pr - nr)/total * np.log(total)
    x.append(rating)

  df['x'] = x

  df = df.drop(rm)
  df = df.reset_index(drop=True)
 
  x = np.array(x)
  y = np.array(y)
  return (df,x,y)