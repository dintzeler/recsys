import logging
import argparse
from .metrics_calculator import DecisionSupportMetrics, ErrorMetrics
from builder.item_similarity_calculator import ItemSimilarityMatrixBuilder
from recs.neighborhood_collaborative_filtering import NeighborhoodBasedRecs
from recs.baseline import BaselineRecommender
import pandas as pd

from recommender.views import Ratings
from django.db.models import Count
import numpy as np

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('Evaluation runner')

class Evaluator(object):

	def __init__(self, builder,
				 recommender,
				 k=10):
		
		self.builder = builder
		self.recommender = recommender
		self.K = k
		

	def calculate(self, min_number_of_ratings, min_rank):
		ratings = load_ratings(min_number_of_ratings)

		users = ratings.user_id.unique()

		#separate users with 80% train and 20% test
		train_data_len = int((len(users) * 80 / 100))
		np.random.seed(42)
		np.random.shuffle(users)
		train_users, test_users = users[:train_data_len], users[train_data_len:]

		#adds to the training test the min_rank ratings of test users
		test_data, train_data = self.split_data(min_rank,
												ratings,
												test_users,
												train_users)	

		logger.debug("Test run having {} training rows, and {} test rows".format(len(train_data), len(test_data)))	

		if self.builder:
			self.builder.build(train_data)

		dsm = DecisionSupportMetrics(self.recommender, self.K)
		ap, ar, map_ = dsm.calculate(train_data, test_data)
		logger.debug("ap:{} ar:{} map:{}".format(ap,ar,map_))

		em = ErrorMetrics(self.recommender)
		mae,rmse = em.calculate(train_data, test_data)
		results = {'ap': ap, 'ar': ar, 'map': map_, 'mae' : mae, 'rmse' : rmse, 'users' : len(users)}
		return results
			





	@staticmethod
	def split_data(min_rank, ratings, test_users, train_users):

		train = ratings[ratings['user_id'].isin(train_users)]

		test_temp = ratings[ratings['user_id'].isin(test_users)]

		test = test_temp.groupby('user_id').head(min_rank)

		additional_training_data = test_temp[~test_temp.index.isin(test.index)]

		train = train.append(additional_training_data)

		return test, train


def evaluate_cf_recommender():

	min_sim = 0.01
	k = 15
	#neighborhood_size = 15
	min_number_of_ratings = 10
	min_overlap = 15
	min_rank = min_number_of_ratings / 2
	for neighborhood_size in range(35,85,10):
		
		recommender = NeighborhoodBasedRecs(neighborhood_size, min_sim)
		er = Evaluator(ItemSimilarityMatrixBuilder(min_overlap, min_sim=min_sim), recommender, k)

		result = er.calculate(min_number_of_ratings, min_rank)
		logger.debug("min_number_of_ratings: {} min_sim: {} k: {} neighborhood_size: {} min_overlap: {}".format(min_number_of_ratings, min_sim,k,neighborhood_size,min_overlap))
		logger.debug("ap: {} ar: {} map: {} mae: {} rmse: {} users: {}".format(result['ap'], result['ar'], result['map'], result['mae'], result['rmse'], result['users']))

def evaluate_bl_recommender():
	min_number_of_ratings = 5
	min_sim = 0.03
	neighborhood_size = 5
	min_rank = min_number_of_ratings / 2
	for k in range(5,105,10):
	
		recommender = BaselineRecommender()
		er = Evaluator(0, recommender, k)
		result = er.calculate(min_number_of_ratings, min_rank)
		logger.debug("min_number_of_ratings: {} min_sim: {} k: {} neighborhood_size: {} ".format(min_number_of_ratings, min_sim,k,neighborhood_size))
		#logger.debug("ap: {} ar: {} map: {} mae: {} rmse: {} users: {}".format(result['ap'], result['ar'], result['map'], result['mae'], result['rmse'], result['users']))
		logger.debug("ap: {} ar: {} map: {}  users: {}".format(result['ap'], result['ar'], result['map'],  result['users']))


def load_ratings(min_ratings=5):

	columns = ['user_id', 'venue_id', 'rating']
		#see which users have rated more than 1 venue
	user_data = Ratings.objects.values('user_id').annotate(intersect = Count('user_id')).filter(intersect__gt = min_ratings) 


	#keep the ratings from those users
	rating_data = Ratings.objects.filter(user_id__in = user_data.values('user_id')).values(*columns)
	#rating_data = Ratings.objects.filter(Q(user_id__in = user_data.values('user_id')) & Q(rating__lte =5)).values(*columns)

	#load the queryset to a pandas dataframe
	ratings = pd.DataFrame.from_records(rating_data, columns=columns)

	return ratings



def main():


	parser = argparse.ArgumentParser(description='Evaluate recommender algorithms.')
	
	
	parser.add_argument('-cf', help="run evaluation on cf rec", action="store_true")

	parser.add_argument('-bs', help="run evaluation on baseline rec", action="store_true")

	args = parser.parse_args()


	if args.cf:
		logger.debug("evaluating cf")
		evaluate_cf_recommender()

	if args.bs:
		logger.debug("evaluating baseline")
		evaluate_bl_recommender()

if __name__ == '__main__':
	main()