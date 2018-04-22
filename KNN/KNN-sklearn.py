import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


#使用KNN预测房价
#wangdm

#前面的数据预处理过程一样
features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
dc_listings = pd.read_csv('listings.csv')
dc_listings = dc_listings[features]
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)#去掉租金前面的美金符号
dc_listings = dc_listings.dropna()  #去掉缺失行
dc_listings[features] = StandardScaler().fit_transform(dc_listings[features])#标准化处理
normalized_listings = dc_listings
norm_train_df = normalized_listings.copy().iloc[0:2792]
norm_test_df = normalized_listings.copy().iloc[2792:]

#使用KNeighborsRegressor
knn = KNeighborsRegressor(50)  #K=50
cols = ['accommodates','bedrooms','bathrooms','beds','minimum_nights','maximum_nights','number_of_reviews']

knn.fit(norm_train_df[cols],norm_train_df['price'])
predict_price = knn.predict(norm_test_df[cols])

mse = mean_squared_error(norm_test_df['price'], predict_price)  #使用sklearn中的评估函数
rmse = mse ** (1/2) #用均方根误差


print(rmse)



