import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# citesc datele din test.csv
df = pd.read_csv('test.csv')

# afisez valorile lipsa
print("Valori lipsă pe coloana:\n", df.isnull().sum())
print("Procent valori lipsa:\n", df.isnull().mean() * 100)

# afisez statisticile descriptive
print("Statistici descriptive:\n", df.describe(include='all'))

# afisez histogramele pentru coloane numerice
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# afisez countplot-ul pentru coloane categorice
plt.figure(figsize=(8,4))
sns.countplot(x='gen', data=df)
plt.title("Distribuția genurilor")
plt.show()

numeric_cols = ['buget', 'durata', 'an_lansare', 'popularitate_director', 'imdb']

# afisez boxplot-uri pentru variabile
for col in numeric_cols:
    plt.figure(figsize=(8, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot pentru {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# afisez heatmap-ul de corelații
plt.figure(figsize=(8,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matrice de corelații")
plt.show()

# afisez distributia valorilor variabilelor in raport cu succesul
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x='succes', y=col, data=df)
    plt.title(f"{col} vs succes")
    plt.xlabel("succes")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

# citesc datele din train.csv
df = pd.read_csv('train.csv')

# afisez valorile lipsa
print("Valori lipsă pe coloana:\n", df.isnull().sum())
print("Procent valori lipsa:\n", df.isnull().mean() * 100)

# afisez statisticile descriptive
print("Statistici descriptive:\n", df.describe(include='all'))

# afisez histogramele pentru coloane numerice
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# afisez countplot-ul pentru coloane categorice
plt.figure(figsize=(8,4))
sns.countplot(x='gen', data=df)
plt.title("Distribuția genurilor")
plt.show()

numeric_cols = ['buget', 'durata', 'an_lansare', 'popularitate_director', 'imdb']

# afisez boxplot-uri pentru variabile
for col in numeric_cols:
    plt.figure(figsize=(8, 3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot pentru {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# afisez heatmap-ul de corelații
plt.figure(figsize=(8,10))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Matrice de corelații")
plt.show()

# afisez distributia valorilor variabilelor in raport cu succesul
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x='succes', y=col, data=df)
    plt.title(f"{col} vs succes")
    plt.xlabel("Succes")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()