import numpy as np
import pandas as pd
import matplotlib as plt

data = pd.read_csv("C:\Users\hp\Desktop\GOOG.csv")
data.head(5)
data.describe()
data = data[["Date","Close"]
data = data.rename(columns = {"Date":"ds","Close":"y"})
data.head(5)
from fbprophet import Prophet
m = Prophet(daily_seasonality = True)
m.fit(data)
future = m.make_future_dataframe(periods=365)
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()