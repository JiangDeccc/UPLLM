# import pandas as pd
# import numpy as np
# import time
# import os

# DIRPATH  = os.path.dirname(__file__)

# def read_log():
# 	file_name=DIRPATH+'/u.data'
# 	df=pd.read_csv(file_name,header=None,sep='\t')
# 	df.columns=['user_id','item_id','rating','timestamp']
# 	# print(df)
# 	return df

# def read_meta():
# 	item_meta=pd.read_csv(DIRPATH+'/u.item',header=None,sep='|', encoding='latin-1')
# 	item_meta.columns=['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

# 	user_meta=pd.read_csv(DIRPATH+'/u.user',header=None,sep='|', encoding='latin-1')
# 	user_meta.columns=['user_id','age','gender','occupation','zip_code']

# 	# print(user_meta,item_meta)
# 	# time.sleep(1000)

# 	return item_meta,user_meta

# if __name__=='__main__':
# 	read_log()
# 	read_meta()

import pandas as pd
import numpy as np
import time
import os
import sys

# DIRPATH  = os.path.dirname(__file__)

def read_log(dataset):
	if dataset == "ml100k":
		train_file = "./ml100k/user_200_3/u_train.data"
		test_file = "./ml100k/user_200_3/u_test.data"
	elif "amazon" in dataset:
		dataset_type = dataset.split("-")[1]
		# train_file = "./amazon/user_200_3/{}/u_train.data".format(dataset_type)
		# test_file = "./amazon/user_200_3/{}/u_test.data".format(dataset_type)
		train_file = "./amazon/user_5k/{}/u_train.data".format(dataset_type)
		test_file = "./amazon/user_5k/{}/u_test.data".format(dataset_type)
	elif dataset == "ml-25m":
		train_file = "./ml-25m/user_5k/u_train.data"
		test_file = "./ml-25m/user_5k/u_test.data"

	df_train = pd.read_csv(train_file, header=None, sep="\t")
	df_train.columns = ['user_id', 'item_id', 'rating', 'timestamp']
	df_test = pd.read_csv(test_file, header=None, sep="\t")
	df_test.columns = ['user_id', 'item_id', 'rating', 'timestamp']
	print("train data shape: {}, test data shape: {}".format(df_train.shape, df_test.shape))
	return df_train, df_test # return all users' rating history

def read_meta(dataset):
	if dataset == "ml100k":
		path = "./ml100k/user_200_3"
		item_meta = pd.read_csv('{}/u.item'.format(path), header=None, sep='|', encoding='latin-1')
	elif "amazon" in dataset:
		dataset_type = dataset.split("-")[1]
		# path = "./amazon/user_200_3/{}".format(dataset_type)
		path = "./amazon/user_5k/{}".format(dataset_type)
		print('{}/u.item'.format(path))
		item_meta = pd.read_csv('{}/u.item'.format(path), header=None, sep='|', quoting=3)
	elif dataset == "ml-25m":
		path = "./ml-25m/user_5k"
		item_meta = pd.read_csv('{}/u.item_new_desc'.format(path), header=None, sep='|', quoting=3)

	item_meta.columns = ["item_id", "name", "category", "description"]
	# print(item_meta.loc[6990])
	user_meta=pd.read_csv('{}/u.user'.format(path),header=None,sep='|', encoding='latin-1')

	if dataset == "ml100k":
		user_meta.columns=['user_id','age','gender','occupation','zip_code']
	elif "amazon" in dataset:
		user_meta.columns = ["user_id", "profile"]
	elif dataset == "ml-25m":
		user_meta.columns = ["user_id", "profile"]
	print("user meta shape: {}, item meta shape: {}".format(user_meta.shape, item_meta.shape))
	return item_meta, user_meta

if __name__=='__main__':
	read_log('amazon-CDs_and_Vinyl')
	read_meta('amazon-CDs_and_Vinyl')