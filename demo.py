import numpy as np
import pandas as pd
from sklearn.svm import SVR
from matplotlib import style
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

# Predefined values
style.use('seaborn')
rcParams['figure.figsize'] = 15, 10


def get_data(filename):
    # Read csv file as panda dataframe
    data = pd.read_csv(filename)
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'MarketCap']
    # Adjast frames
    data['Date'] = pd.to_datetime(data['Date'])
    data['Average'] = data.eval('Open + Close') / 2
    # Adjust diff column
    data['Diff'] = data['Average'] - data['Average'].shift(1)
    # Return data as lists
    return data['Date'].tolist()[1:], data['Average'].tolist()[1:], data['Diff'].tolist()[1:]


def predict_price(indices, dates, prices, x):
    # Converting to matrix of n X 1
    print('>>>>> STATUS: MATRIX CONVERSION')
    indices = np.reshape(indices,(len(indices), 1))
    # Defining the support vector regression models
    print('>>>>> STATUS: DEFINING SVR MODEL')
    # Defining the support vector regression models
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1)
    # Fitting the data points in the models
    svr_rbf.fit(indices, prices)
    # Plotting the results
    print('>>>>> STATUS: PLOTTING')
    plt.scatter(dates, prices, color= 'black', label= 'Data')
    plt.plot(dates, svr_rbf.predict(indices), color= 'red', linewidth=2, label= 'RBF model')
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    print('>>>>> STATUS: MAKING A PREDICTION')
    return svr_rbf.predict(x)


def delete_outlayers(input_list, price_diff):
    max_pos = price_diff.index(max(price_diff))
    min_pos = price_diff.index(min(price_diff))
    input_list.remove(input_list[max_pos])
    input_list.remove(input_list[min_pos])


# Start and import data
print('>>>>> START')
dates, prices, price_diff = get_data('data.csv')
indices = [i for i in range(0, len(prices))]

# Quick filtering to enable good visualization
# This section can be commented if wished
delete_outlayers(dates, price_diff)
delete_outlayers(prices, price_diff)
delete_outlayers(indices, price_diff)
price_diff.remove(max(price_diff))
price_diff.remove(min(price_diff))
print('>>>>> STATUS: DATA FORMATTING DONE')

# Model and prediction
predicted_price = predict_price(indices, dates, price_diff, 1829)
print ('RESULTING PREDICTION = ', (predicted_price*-1)+prices[0])
print('>>>>> DONE')

