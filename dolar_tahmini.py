# kutuphanelerin cagrilmasi

import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas_datareader import data as pdr
import yfinance as yf

'''
baslangic ve bitis tarihlerini asagidan duzeltebilirsiniz ayrica egitimin kac defa yapilacagini da alttaki "egitim_sayisi" degiskenin
degerini degistirerek belirtebilirsiniz!
"ticker" degiskenin degerini de yahoo finance sitesinde doviz kisaltmalariyla degistirerek farkli islemlerde yapabilirsiniz.
'''
ticker = "USDTRY=X"
start = dt.datetime(2021, 1, 1)
end = dt.datetime.now()
egitim_sayisi = 1

#dolar Turk Lirasi verilerinin yahoo finance sitesinde cekilmesi
df = yf.download(ticker, start_date, end_date)

data = df.filter(['Close'])

#datayi numpy serisine donustur

dataset = data.values

#egitilecek satirlari al

training_data_len = math.ceil(len(dataset)* .8)

#datayi olceklendir

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#egitilecek satirlari yarat

train_data = scaled_data[0:training_data_len, :]

#egitilmis datayi x ve t koordinatlarinada ayir

x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

# x_train ve y_train modellerini numpy serilerine cevir

x_train, y_train = np.array(x_train), np.array(y_train)

#datayi yeniden sekillendir

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#LSTM model yapilandirmasi

model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape =(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#fiyat modellemesi derle

model.compile(optimizer='Adam', loss='mean_squared_error')

#modeli egit

model.fit(x_train, y_train, batch_size=1, epochs=egitim_sayisi)

#test et

test_data = scaled_data[training_data_len - 60:,:]

# x_test ve y_test datasetini yarat

x_test = []
y_test = dataset[training_data_len:,:]
for y in range(60,len(test_data)):
  x_test.append(test_data[y-60:y, 0])

#datayi numpy serisine cevir

x_test = np.array(x_test)

#datayi 3 boyutluya sekillendir

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#tahmin edilen fiyatlari al

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#RMSE

rmse = np.sqrt(np.mean(predictions - y_test)**2)

#veriyi tasi

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Tahminler'] = predictions

#veriyi gorsellestir

plt.figure(figsize=(10,5))
plt.title('Model')
plt.xlabel('Tarih', fontsize=18)
plt.ylabel('Kapanis Fiyatlari', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Tahminler']])
plt.legend(['Egitilen Alan', 'Deger', 'Tahminler'], loc='lower right')
plt.show()

#tahmin edilen fiyatla gecerli fiyatlari goster

print(valid)








