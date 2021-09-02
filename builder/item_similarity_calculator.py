import django
import os
from datetime import datetime
from scipy.sparse import coo_matrix
import pandas as pd
from django.db.models import Count
import psycopg2
import logging
from tqdm import tqdm
import math

from sklearn.metrics.pairwise import cosine_similarity
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "recsys.settings")
django.setup()
from recsys import settings
from recommender.models import Ratings, Similarity

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger('Item simialarity calculator')


class ItemSimilarityMatrixBuilder(object):

	def __init__(self, min_overlap=15, min_sim=0.01):
		self.min_overlap = min_overlap
		self.min_sim = min_sim
		self.db = settings.DATABASES['default']['ENGINE']
	
	def build(self, ratings):
		logger.debug("Calculating similarities ... using {} ratings".format(len(ratings)))
		start_time = datetime.now()


		ratings['rating'] = ratings['rating'].astype(float)
		ratings['avg'] = ratings.groupby('user_id')['rating'].transform(lambda x: normalize(x))
		ratings['user_id'] = ratings['user_id'].astype('category')
		ratings['venue_id'] = ratings['venue_id'].astype('category')



		
		#create the coo sparse matrix 
		coo = coo_matrix((ratings['avg'].astype(float),(ratings['venue_id'].cat.codes.copy(),ratings['user_id'].cat.codes.copy())))

		logger.debug("Calculating overlaps between the items")
		overlap_matrix = coo.astype(bool).astype(int).dot(coo.transpose().astype(bool).astype(int))

		number_of_overlaps = (overlap_matrix > self.min_overlap).count_nonzero()
		logger.debug("Overlap matrix leaves {} out of {} with {}".format(number_of_overlaps,
                                                                         overlap_matrix.count_nonzero(),
                                                                         self.min_overlap))
        
		logger.debug("Rating matrix (size {}x{}) finished, in {} seconds".format(coo.shape[0],
                                                                                 coo.shape[1],
                                                                                 datetime.now() - start_time))

		sparsity_level = 1 - (ratings.shape[0] / (coo.shape[0] * coo.shape[1]))
		logger.debug("Sparsity level is {}".format(sparsity_level))

		venues = dict(enumerate(ratings['venue_id'].cat.categories))

		cor = cosine_similarity(coo, dense_output=False)
		cor = cor.multiply(cor > self.min_sim)
		cor = cor.multiply(overlap_matrix > self.min_overlap)
		logger.debug('Correlation is finished, done in {} seconds'.format(datetime.now() - start_time))		


		start_time = datetime.now()
		self._save_similarities(cor, venues)
		logger.debug('save finished, done in {} seconds'.format(datetime.now() - start_time))
		return cor, venues

	def _save_similarities(self, sm, index):
		start_time = datetime.now()

		logger.debug('truncating table in {} seconds'.format(datetime.now() - start_time))
		sims = []
		no_saved = 0
		start_time = datetime.now()
		coo = coo_matrix(sm)
		csr = coo.tocsr()

		logger.debug('instantiation of coo_matrix in {} seconds'.format(datetime.now() - start_time))

		query = "insert into similarity (source, target, similarity) values %s;"

		conn = self._get_conn()

		cur = conn.cursor()

		cur.execute('truncate table similarity')

		logger.debug('{} similarities to save'.format(coo.count_nonzero()))
		xs, ys = coo.nonzero()
		for x, y in tqdm(zip(xs, ys), leave=True):

			if x == y:
				continue

			sim = csr[x, y]

			if sim < self.min_sim:
				continue

			if len(sims) == 500000:
				psycopg2.extras.execute_values(cur, query, sims)
				sims = []
				logger.debug("{} saved in {}".format(no_saved,
                                                     datetime.now() - start_time))

			new_similarity = (index[x], index[y], sim) #get the venues_id from categories x,y
			no_saved += 1
			sims.append(new_similarity)
		
		psycopg2.extras.execute_values(cur, query, sims, template=None, page_size=1000)
		conn.commit()
		logger.debug('{} Similarity items saved, done in {} seconds'.format(no_saved, datetime.now() - start_time))





	@staticmethod
	def _get_conn():
		dbUsername = settings.DATABASES['default']['USER']
		dbPassword = settings.DATABASES['default']['PASSWORD']
		dbName = settings.DATABASES['default']['NAME']
		conn_str = "dbname={} user={} password={}".format(dbName, dbUsername, dbPassword)
		conn = psycopg2.connect(conn_str)

		return conn

def normalize(x):
	x = x.astype(float)
	x_sum = x.sum()
	x_num = x.astype(bool).sum()
	x_mean = x_sum / x_num

    #if x.std() == 0 then the normalized rating tends to infinity. This means that the user has rated each venue equally
	if x_num == 1 or x.std() < 0.00000001:
		return 0.0
	return (x - x_mean) / (x.max() - x.min())


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

	logger.info("Calculation of item similarity")

	all_ratings = load_all_ratings()

	ItemSimilarityMatrixBuilder(min_overlap=15, min_sim=0.01).build(all_ratings)
