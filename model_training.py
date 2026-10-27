import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# citesc datele
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# fac one-hot encoding
train_df = pd.get_dummies(train_df, columns=['gen'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['gen'], drop_first=True)

# separ features si eticheta
X_train = train_df.drop('succes', axis=1)
y_train = train_df['succes']
X_test = test_df.drop('succes', axis=1)
y_test = test_df['succes']

# fac standardizarea valorilor
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# antrenez modelul
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# fac evaluarea pe test.csv
y_pred = model.predict(X_test)
print("Acuratete pe test.csv\n", accuracy_score(y_test, y_pred))
print("Raport clasificare test.csv\n", classification_report(y_test, y_pred))

# afisez matricea de confuzie
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Matrice de confuzie (test.csv)")
plt.xlabel("Valori prezise")
plt.ylabel("Valori reale")
plt.show()