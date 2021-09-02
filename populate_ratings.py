import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recsys.settings')
import django
django.setup()
import math

import pandas as pd 
import tqdm
import numpy
from recommender.models import Ratings, Log

def create(model, **kwargs):
    # Actual Django statement:
    return model.objects.create(**kwargs)



def populate_ratings():
	ratings_preprocess = pd.DataFrame.from_records(Log.objects.all().values())
	ratings_preprocess = ratings_preprocess.groupby(by=['user_id','venue_id','year'], axis=0, as_index = False).sum()

	ratings_preprocess = ratings_preprocess.groupby(by = ['user_id','venue_id'], axis=0, as_index = False, group_keys = False).apply(lambda x: sum((1+x.year-x.year.min())*numpy.log10(x.number_of_times+1)))
	df = ratings_preprocess.index.to_frame(index = False)			
	df = pd.DataFrame({'user_id':df["user_id"],'venue_id':df["venue_id"],'rating':ratings_preprocess.values[:,2]})
		

	for index, row in tqdm.tqdm(df.iterrows()):
		user_id = row['user_id']
		venue_id = row['venue_id']
		create(Ratings, user_id = user_id, venue_id = venue_id, rating = row['rating'] )

if __name__ == '__main__':
	populate_ratings()