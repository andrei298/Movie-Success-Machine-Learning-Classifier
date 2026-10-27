import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
n_total = 2400
genuri = ['actiune', 'comedie', 'drama', 'horror', 'science-fiction', 'romantic']

data = {

    'gen': np.random.choice(genuri, size=n_total),
    'durata': np.random.randint(60, 240, size=n_total),
    'an_lansare': np.random.randint(1920, 2024, size=n_total),
    'buget': np.round(np.clip(np.random.normal(loc=40, scale=15, size=n_total), 1, None), 2),
    'actor_faimos': np.random.choice([0, 1], size=n_total, p=[0.8, 0.2]),
    'popularitate_director': np.round(np.random.uniform(0.0, 1.0, size=n_total), 2),
    'imdb': np.round(np.random.uniform(1.0, 9.5, size=n_total), 1),

}

data['succes'] = (
    ((data['imdb'] > 7.5)) | ((data['buget'] > 30) & 
    (data['popularitate_director'] > 0.7) & (data['actor_faimos'] == 1) & (data['durata'] > 90) & (data['durata'] < 180))
).astype(int)

df = pd.DataFrame(data)
train_df, test_df = train_test_split(df, test_size=800, random_state=42)
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("train.csv și test.csv au fost generate.")