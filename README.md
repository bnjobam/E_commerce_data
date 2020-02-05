# E_commerce_data
Analysis if ecommerce data

analysis_ecommerce.py presents the preparation and analysis of ecommerce data gotten from kaggle: 
https://www.kaggle.com/retailrocket/ecommerce-dataset/download
containing events: visitorid, itemid, timestamp event(view, addtocart, transaction)
item_properties: itemid, timestamp, property, value
and category_table.
To write a model that predicts the items in the addtocart event from items in view event, I generated new variables
number of times a particular item is viewed by a visitor and if the item is availabilty of the item when it is being
viewed (which was used to remove useless data points)
From here I build an extreme boosted trees model and got an accuracy of 0.79.
The second task is to determine abnormal e-shoppers who generate useless information that hinders algorithms like
recommender systems. With this I developed a formula to determine useful shopper information based on common sense
that can be applied to business logic.
Generally if a shopper seriously considers a product, s\he is more likely to view it for more than once (the most signicant
predictor for addtocart was number of views of an item by the shopper) thus if a shopper views an item once, the information 
for that shopper is useless. One determinant I used was the average number of views per day by a shopper for the time interval
between the first view and the last, a business can determine their cut for the number of views 400-500 seems optimal but in the 
code I used 20.
