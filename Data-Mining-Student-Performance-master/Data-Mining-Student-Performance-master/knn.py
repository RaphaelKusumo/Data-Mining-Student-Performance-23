import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#medu = waktu belajar
#nilai_tes = hasil ujian
data = { 
    'sex': ['wanita', 'wanita', 'wanita', 'wanita', 'pria',' pria'],
    'age' : [18, 17, 15, 15,16, 16],
    'medu': [4, 3.5, 1, 2, 3, 1.5],
    'nilai_tes': [8, 8, 5, 5, 7, 6],
    'hasil' : ['lulus', 'lulus', 'tidak lulus', 'tidak lulus', 'lulus', 'tidak lulus']
}
data_df = pd.DataFrame(data)
print(data_df)

#visualisasi Model
fig, ax = plt.subplots()
for hasil, d in data_df.groupby('hasil') :
    ax.scatter( d['medu'],d['nilai_tes'], label = hasil)

plt.legend(loc='upper left')
plt. title('sebaran data')
plt.xlabel('Waktu belajar')
plt.ylabel('Hasil nilai Ujian')
plt.grid(True)
plt.show()

#Preprocessing dataset 
x_train = np.array(data_df[['medu','nilai_tes']])
y_train = np.array(data_df['hasil'])

print(f'x_train:\n{x_train}\n')
print(f'y_train: {y_train}')

from sklearn.preprocessing import LabelBinarizer, label_binarize
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
print(f'y_train: {y_train}')

y_train = y_train.flatten()
print(f'y_train: {y_train}')

#training KNN Model
from sklearn.neighbors import KNeighborsClassifier
K = 3
model = KNeighborsClassifier(n_neighbors=K)
model.fit(x_train,y_train)

#prediksi kelulusan
medu = 3
nilai_tes = 8
x_new = np.array([medu, nilai_tes]).reshape(1, -1)
print(f'x_new: {x_new}')

y_new = model.predict(x_new)
print(f'y_new: {y_new}')

y_new = lb.inverse_transform(y_new)
print(f'y_new: {y_new}')
