import time
import json
import re
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

# def json_to_csv(directory, fileNames, createSample=False):
#     """
#     json_to_csv: loops through specified JSON files and converts them to csv files.
#                  option to also create a sample csv, which uses np.random.seed 9001 to create a sample dataset with 10% of the observations
#
#                  pandas has a read_json function, but returns a 'Trailing data error' when working with these specific files
#
#
#     Inputs: -directory of JSON files
#             -list of JSON filenames
#             -createSample flag
#
#     """
#
#     start = time.time()
#
#     jsonData = []
#
#     for fileName in fileNames:
#         with open(directory + fileName,  encoding="utf8") as file:
#             print('{0} opened'.format(fileName))
#             for line in file:
#                 #I use an rstrip here because some of the files have trailing blank spaces
#                 jsonData.append(json.loads(line.rstrip()))
#
#         df = pd.DataFrame.from_dict(jsonData)
#
#         csvFileName = fileName[:len(fileName)-5] + '.csv'
#
#         df.to_csv(directory + csvFileName)
#         print('{0} created'.format(csvFileName))
#
#
#         if createSample:
#             np.random.seed(9001)
#             msk = np.random.rand(len(df)) <= 0.1
#             sample = df[msk]
#
#             csvSampleFileName = fileName[:len(fileName)-5] + '_sample.csv'
#
#             sample.to_csv(directory + csvSampleFileName)
#             print('{0} created'.format(csvSampleFileName))
#
#     print('This function took {} minutes to run'.format((time.time()-start)/60))
#
#
# # fileNameList = ['user.json',
# #                 'business.json',
# #                 'review.json']
#
# # json_to_csv('data/', fileNameList, createSample=True)
#
# df_business = pd.read_json('data/business.json', lines=True)
# df_business.dropna(inplace=True, subset = ['categories'], axis=0)
# df_business.loc[df_business['categories'].str.contains('Restaurants')]
# df_business['categories'].value_counts()
#
# df_business.head()
#
# df_business.count()

# # reading filter_review
df_review = pd.read_csv('data/filtered_reviews.csv', index_col=0)
df_review.dropna
#
# inumerating business_in and user_id with bid and uid
def build_fmap_invmap(ser):
    uni_ele = ser.unique()
    fmap = {v:i for i, v in enumerate(uni_ele)}
    invmap = {i:v for i, v in enumerate(uni_ele)}
    return fmap, invmap

# setting debuging enviroment on (dbg =1) to turn it off (dbg = 0)
dbg = 1
if dbg:
    df_review = df_review.head(100000)

bus_fmap, bus_invmap = build_fmap_invmap(df_review['business_id'])
u_fmap, u_invmap = build_fmap_invmap(df_review['user_id'])

df_review['bid'] = df_review['business_id'].map(bus_fmap)

df_review['uid'] = df_review['user_id'].map(u_fmap)

df_review.head()

n_users, n_bus = df_review['uid'].nunique(), df_review['bid'].nunique()

n_dim = 5

# Initializing tensor flow at a randon number | n_users * n_dim (layers) initializing at some random number between
# -1 to 1 both for business and users. For internal layer.
# PS > Create a function with code below:

user_vector_raw = tf.Variable(tf.random_uniform([n_users, n_dim], minval = -1., maxval = 1.))
bus_vector_raw = tf.Variable(tf.random_uniform([n_bus, n_dim], minval = -1., maxval = 1.))

# running the tanh function to find
user_vector = tf.tanh(user_vector_raw)
bus_vector = tf.tanh(bus_vector_raw)

# Stipulating the imput layer.
users = tf.placeholder(tf.int32, shape=(None))
businesses = tf.placeholder(tf.int32, shape=(None))
ratings = tf.placeholder(tf.float32, shape=(None))

UserSampled = tf.nn.embedding_lookup(user_vector, users)
BusinessSampled = tf.nn.embedding_lookup(bus_vector, businesses)
UserSampled.set_shape([None, n_dim])
BusinessSampled.set_shape([None, n_dim])

# input tensors for products, users, ratings

# Defining the output
# transfer into a fucntion
estimatedaffinitiesraw = tf.reduce_sum(UserSampled * BusinessSampled, 1)
estimatedaffinities = tf.sigmoid(estimatedaffinitiesraw)*5

# estimatedaffinities - ratings ask Lee to clarify ratings, where that ratings comes from? ask to explain the loss function
# transfer into a function
loss = tf.reduce_sum(tf.square(estimatedaffinities - ratings))
opt = tf.train.RMSPropOptimizer(learning_rate=.1).minimize(loss)

# Setting the session and intialize it

sess = tf.Session()

# picking up 64 randon rows in order to run under memory capacity
rows = np.random.choice(df_review.shape[0], 64)

sess.run(tf.global_variables_initializer())

# Creating a loop to train under 64 random rows
for i in range(10000):
    rows = np.random.choice(df_review.shape[0], 64)
    dfrows = df_review.iloc[rows]
    fd = {users:dfrows['uid'].values,
         businesses:dfrows['bid'].values,
         ratings:dfrows['stars'].values}
    _, l2loss = sess.run([opt, loss], fd)
    if i % 1000 == 0:
        print(l2loss)

user_values, bus_values = sess.run([user_vector, bus_vector])

bus_vec_df = pd.DataFrame(data = bus_values, index =
                          [bus_invmap[i] for i in range(n_bus)])



bus_vec_df

# Joining df_business + bus_vec_df
# df_allBusiness = df_business.join(bus_vec_df, on='business_id', how='right')
#
# df_allBusiness = df_allBusiness.dropna()

# Pulling user ID 4 and comparing to inverse map on uid (not sure why, maybe to check accuracy?)
# uid = 4
# u_invmap[uid]
#
#
# bname = 'cHdJXLlKNWixBXpDwEGb_A'
# bid = bus_fmap[bname]
#
# df_allBusiness
#
# df_allBusiness.count()
#
# df_allBusiness.loc[df_allBusiness['categories'].str.contains('Restaurant') &
#            df_allBusiness['categories'].str.contains('Japanese')]
#
# # Testing | Passing Train
#
# bid
#
# chipotle = bus_values[bid]
#
# chipotle
#
# japaneselover = user_values[uid]
#
# bus_values
#
# np.square(bus_values - japaneselover[None,:]).sum(1).argsort()
#
# np.square(bus_values - chipotle[None,:]).sum(1).argsort()
#
def closest_businesses_to(business = None, user = None, df = None):
    if business is not None:
        target = bus_values[bus_fmap[business]]
    if user is not None:
        target = user_values[u_fmap[user]]
    if df is None:
        df = bus_values
    best_restaurants = np.square(df - target[None,:]).sum(1).argsort()
    return best_restaurants
#
# midtown_japanese_restaurants = bus_values[:30,:]
#
# closest_businesses_to(business = 'cHdJXLlKNWixBXpDwEGb_A')
#
#
# closest_businesses_to(user = 'ri7itn7-CdpsaPxTToK5cQ')
#
# closest_businesses_to(user = 'ri7itn7-CdpsaPxTToK5cQ', df = midtown_japanese_restaurants)

### Saving train









# Dropping NaN

df_userSample = pd.read_csv('data/user_sample.csv')
df_userSample = df_userSample.dropna()

df_userSample.count()

df_reviewSample = pd.read_csv('data/review_sample.csv')
df_reviewSample = df_reviewSample.dropna()

df_reviewSample.count()

def find_ftres_with_nan(df):
    all_nan = df.columns[df.isnull().all()].tolist()
    some_nan = df.columns[df.isnull().any()].tolist()
    print("All NaN Features: ", len(all_nan), all_nan, "Some NaN Features: ", len(some_nan), some_nan)
    return all_nan, some_nan

business = pd.read_csv('data/business.csv',encoding = "ISO-8859-1",index_col=0)
all_nan, some_nan = find_ftres_with_nan(business)

### Number of businesses that have both "food" and "restaurant" in their category:


# create a mask for restaurants

mask_restaurants = business['categories'].str.contains('Restaurants')

# create a mask for food
mask_food = business['categories'].str.contains('Food')

# apply both masks
restaurants_and_food = business[mask_restaurants & mask_food]

# number of businesses that have food and restaurant in their category
restaurants_and_food['categories'].count()

### Even after taking buisnesses that have both food and restaurant in their categories, there are still irrelevant business categories in the data.

# an example row
restaurants_and_food.head(1)['categories'].values

### Thus, we manually identified additional categories that needed to be excluded specifically.

categoryDF = restaurants_and_food['categories'].apply(lambda x: x[1:-1].split(',')).apply(pd.Series)
uniqueCategories = pd.DataFrame(categoryDF.stack().str.strip().unique())

categoriesToRemove = ['Grocery','Drugstores','Convenience Stores','Beauty & Spas','Photography Stores & Services',
                      'Cosmetics & Beauty Supply','Discount Store','Fashion','Department Stores','Gas Stations',
                      'Automotive','Music & Video','Event Planning & Services','Mobile Phones','Health & Medical',
                      'Weight Loss Centers','Home & Garden','Kitchen & Bath','Jewelry',"Children's Clothing",
                      'Accessories','Home Decor','Bus Tours','Auto Glass Services','Auto Detailing',
                      'Oil Change Stations', 'Auto Repair','Body Shops','Car Window Tinting','Car Wash',
                      'Gluten-Free','Fitness & Instruction','Nurseries & Gardening','Wedding Planning',
                      'Embroidery & Crochet','Dance Schools','Performing Arts',
                      'Wholesale Stores','Tobacco Shops','Nutritionists','Hobby Shops','Pet Services',
                      'Electronics','Plumbing','Gyms','Yoga','Walking Tours','Toy Stores','Pet Stores',
                      'Pet Groomers','Vape Shops','Head Shops',
                      'Souvenir Shops','Pharmacy','Appliances & Repair','Wholesalers','Party Equipment Rentals',
                      'Tattoo','Funeral Services & Cemeteries','Sporting Goods','Dog Walkers',
                      'Pet Boarding/Pet Sitting','Scavenger Hunts','Contractors','Trainers',
                      'Customized Merchandise', 'Dry Cleaning & Laundry', 'Art Galleries'
                      'Tax Law', 'Bankruptcy Law', 'Tax Services', 'Estate Planning Law',
                      'Business Consulting', 'Lawyers', 'Pet Adoption', 'Escape Games',
                      'Animal Shelters', 'Commercial Real Estate', 'Real Estate Agents',
                      'Real Estate Services', 'Home Inspectors']


restaurants_df = restaurants_and_food[~restaurants_and_food['categories'].str.contains('|'.join(categoriesToRemove))]

restaurants_df.to_csv('data/restaurants.csv')
restaurants_df = pd.read_csv('data/restaurants.csv', encoding='ISO-8859-1', index_col=0)
restaurants_df = restaurants_df.dropna(axis=1)

restaurants_df.head()

# Building

#task 1
def get_restaurants(keyword):
    return restaurants_df.loc[restaurants_df['categories'].str.contains(keyword)]

get_restaurants('Japanese')

#task 2
def get_reviews_for(rest_id):
    return df_reviewSample.loc[df_reviewSample['business_id']==rest_id]

get_reviews_for('19fdSca3MUoaGFNX2BrjTQ')


import pdb

#task 3
def get_recommendations_for(user_id = None, business_id = None):
    if user_id is not None:
        bids = closest_businesses_to(user = user_id)
    else:
        bids = closest_businesses_to(business = business_id)
    bnames = [bus_invmap[b] for b in bids]
    return restaurants_df.set_index('business_id').loc[
        [b for b in bnames if b in restaurants_df['business_id'].values]]#.dropna()

get_recommendations_for(business_id = 'a7mTbEi2N8Zd-r-8jlReww')

get_recommendations_for(user_id= '96s7b2PBjmkzEeQTzmKp7w')

### Building Geo Table

restaurantsGeo_df = restaurants_df.drop(['is_open', 'review_count'], 1)

lat = 44
lon = -70
distance = np.sqrt((restaurantsGeo_df['latitude'] - lat)**2 + (restaurantsGeo_df['longitude'] - lon)**2)

restaurantsGeo_df.loc[distance < 5]

#task 4
def filter_by_location(df, lat, lon, max_distance):
    distance = np.sqrt((df['latitude'] - lat)**2 + (df['longitude'] - lon)**2)
    return df.loc[distance < max_distance]






def get_recommendations_for_locally(user_id = None, business_id = None, lat = 0, lon = 0, max_distance = 100):
    if user_id is not None:
        bids = closest_businesses_to(user = user_id)
    else:
        bids = closest_businesses_to(business = business_id)
    bnames = [bus_invmap[b] for b in bids]
    closest_businesses = restaurants_df.set_index('business_id').loc[
        [b for b in bnames if b in restaurants_df['business_id'].values]]#.dropna()
    filtered_by_location = filter_by_location(closest_businesses, lat, lon, max_distance)
    return filtered_by_location

def filter_by_keyword(df, keyword_include = None, keyword_exclude = None):
    if keyword_include is not None:
        #throw away restaurants who don't include keyword
        df = df.loc[df['categories'].str.contains(keyword_include)]
    if keyword_exclude is not None:
        #throw away restaurants who include keyword
        df = df.loc[~df['categories'].str.contains(keyword_exclude)]
    return df


def get_recommendations_for_locally_by_keyword(
        user_id = None, business_id = None, lat = 0, lon = 0, max_distance = None,
       keyword_include = None, keyword_exclude = None):
    if user_id is not None:
        bids = closest_businesses_to(user = user_id)
    else:
        bids = closest_businesses_to(business = business_id)
    bnames = [bus_invmap[b] for b in bids]
    df = restaurants_df.set_index('business_id').loc[
        [b for b in bnames if b in restaurants_df['business_id'].values]]#.dropna()
    if max_distance is not None:
        df = filter_by_location(df, lat, lon, max_distance)
    if keyword_include is not None or keyword_exclude is not None:
        df = filter_by_keyword(df, keyword_include, keyword_exclude)
    return df

get_recommendations_for_locally_by_keyword(
    user_id= '96s7b2PBjmkzEeQTzmKp7w', lat = 44, lon = -70, max_distance = 10,
keyword_include = 'Korean',keyword_exclude = 'Sushi')




def get_recommendations_for_locally(user_id = None, business_id = None, lat = 0, lon = 0, max_distance = 100):
    if user_id is not None:
        bids = closest_businesses_to(user = user_id)
    else:
        bids = closest_businesses_to(business = business_id)
    bnames = [bus_invmap[b] for b in bids]
    closest_businesses = restaurants_df.set_index('business_id').loc[
        [b for b in bnames if b in restaurants_df['business_id'].values]]#.dropna()
    filtered_by_location = filter_by_location(closest_businesses, lat, lon, max_distance)
    return filtered_by_location

get_recommendations_for_locally(business_id = 'a7mTbEi2N8Zd-r-8jlReww', lat = 44, lon = -70, max_distance = 10)

get_recommendations_for_locally(user_id= '96s7b2PBjmkzEeQTzmKp7w')
