import os
import json
import time
import argparse
import logging
from decimal import Decimal
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys.settings")

import django

django.setup()
from django.db.models import Count
from recommender.models import Ratings
from recs.neighborhood_collaborative_filtering import NeighborhoodBasedRecs


class RecommenderCoverage(object):

	def __init__(self, recommender):
		self.ratings = self.load_all_ratings()
		self.all_users = set(self.ratings['user_id'])
		self.all_venues = set(self.ratings['venue_id'])
		self.recommender = recommender
		self.items_in_rec = defaultdict(int)
		self.no_users_with_recs = 0

	def calculate_coverage(self, K=15):

		logger.debug('calculating coverage for all users ({} in total)'.format(len(self.all_users)))
		
		for user in tqdm(list(self.all_users)):
			recset = self.recommender.recommend_items(int(user), num=K)
			if recset:
				
				self.no_users_with_recs +=1 
				for rec in recset:
					self.items_in_rec[rec[0]] += 1
					

		

		no_venues = len(self.all_venues)
		no_venues_in_rec = len(self.items_in_rec)
		no_users = len(self.all_users)
		user_coverage = float(self.no_users_with_recs / no_users)
		venue_coverage = float(no_venues_in_rec / no_venues)
		logger.info("{} {} {}".format(no_users, self.no_users_with_recs, user_coverage))
		logger.info("{} {} {}".format(no_venues, no_venues_in_rec, venue_coverage))
		return user_coverage, venue_coverage


	@staticmethod
	def load_all_ratings(min_ratings=1):

		columns = ['user_id', 'venue_id', 'rating']

		#see which users have rated more than 1 venue
		user_data = Ratings.objects.values('user_id').annotate(intersect = Count('user_id')).filter(intersect__gt = min_ratings) 

		#keep the ratings from those users
		rating_data = Ratings.objects.filter(user_id__in = user_data.values('user_id')).values(*columns)

		#load the queryset to a pandas dataframe
		ratings = pd.DataFrame.from_records(rating_data, columns=columns)
		return ratings


if __name__ == '__main__':
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
	logger = logging.getLogger('Evaluation runner')
	parser = argparse.ArgumentParser()
	parser.add_argument('-cf', help="run evaluation on cf rec", action="store_true")
	args = parser.parse_args()
	
	if args.cf:
		logger.debug("evaluating coverage of cf")
		cov = RecommenderCoverage(NeighborhoodBasedRecs())
		cov.calculate_coverage(K=15)
	
	