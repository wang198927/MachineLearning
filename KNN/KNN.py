import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

#使用KNN预测房价
#wangdm


def predict_price(new_listing,feature_columns):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return(predicted_price)

features = ['accommodates','bedrooms','bathrooms','beds','price','minimum_nights','maximum_nights','number_of_reviews']
dc_listings = pd.read_csv('listings.csv')
dc_listings = dc_listings[features]
dc_listings['price'] = dc_listings.price.str.replace("\$|,",'').astype(float)#去掉租金前面的美金符号
dc_listings = dc_listings.dropna()  #去掉缺失行
dc_listings[features] = StandardScaler().fit_transform(dc_listings[features])#标准化处理
normalized_listings = dc_listings
norm_train_df = normalized_listings.copy().iloc[0:2792]
norm_test_df = normalized_listings.copy().iloc[2792:]

cols = ['accommodates','bedrooms','bathrooms','beds','minimum_nights','maximum_nights','number_of_reviews']
norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price,feature_columns = cols,axis =1)
norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - norm_test_df['price'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2) #用均方根误差
print(rmse)



