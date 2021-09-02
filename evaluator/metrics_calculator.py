from decimal import Decimal
from tqdm import tqdm
from math import sqrt
class DecisionSupportMetrics(object):

	def __init__(self, recommender, k):
		self.K = k 
		self.recommender = recommender

	def calculate(self, train_ratings, test_ratings):
		total_precision_score = Decimal(0.0)
		total_recall_score = Decimal(0.0)
		total_precision_score_alt = Decimal(0.0)
		ap = []
		ar = []
		map_ = []

		for user_id, users_test_data in tqdm(test_ratings.groupby('user_id')):
			#user_id_count += 1

			#the data of test user_id that are in traing_data
			training_data_for_user = train_ratings[train_ratings['user_id'] == user_id]
			dict_for_rec = training_data_for_user.to_dict(orient='records')

			#relevant ratings of user_id
			relevant_ratings = list(users_test_data['venue_id'])
			if len(dict_for_rec) > 0:
				recs = list(self.recommender.recommend_items_by_ratings(user_id, dict_for_rec, num=self.K))
				if len(recs)>0:
					precision = self.precision_at_k(recs, relevant_ratings)
					recall = self.recall_at_k(recs, relevant_ratings)
					alt_precision = self.average_precision_k(recs, relevant_ratings)
					ap.append(precision)
					ar.append(recall)
					map_.append(alt_precision)
					total_precision_score += precision
					total_recall_score += recall
					total_precision_score_alt += alt_precision


		average_recall = total_recall_score/len(ar) if len(ar) > 0 else 0
		average_precision = total_precision_score/len(ap) if len(ap) > 0 else 0
		mean_average_precision = total_precision_score_alt/len(map_)
		return average_precision, average_recall, mean_average_precision

			
	@staticmethod
	def recall_at_k(recs, relevant_ratings):
		if len(relevant_ratings) == 0:
			return Decimal(0.0)

		TP = set([r[0] for r in recs if r[0] in relevant_ratings])

		return Decimal(len(TP) / len(relevant_ratings))

	@staticmethod
	def precision_at_k(recs, relevant_ratings):
		if len(relevant_ratings) == 0:
			return Decimal(0.0)

		TP = set([r[0] for r in recs if r[0] in relevant_ratings])

		return Decimal(len(TP) / len(recs))

	@staticmethod
	def average_precision_k(recs, relevant_ratings):
		score = Decimal(0.0)
		num_hits = 0

		for i, p in enumerate(recs):
			TP = p[0] in relevant_ratings
			if TP:
				num_hits += 1.0
			score += Decimal(num_hits / (i + 1.0))
		if score > 0:
			score /= min(len(recs), len(relevant_ratings))
		return score

class ErrorMetrics(object):

	def __init__(self, recommender):
		self.recommender = recommender


	def calculate(self, train_ratings, test_ratings):

		number_of_users = len(test_ratings['user_id'].unique())
		mae = Decimal(0.0)
		rmse = Decimal(0.0)
		n_venues_rmse = Decimal(0.0)
		for user, user_data in tqdm(test_ratings.groupby('user_id')):
			sum_mae = Decimal(0.0)

			ratings_for_rec = train_ratings[train_ratings['user_id']==user]
			user_training_venues = {v['venue_id']: Decimal(v['rating']) for v in ratings_for_rec[['venue_id', 'rating']].to_dict(orient='records')}			
			
			venues_test_user = user_data['venue_id'].tolist()
			n_venues = 0

			#if the venues of the user in test_data is zero then the error of the user is 0
			if len(venues_test_user)>0:

				for venue in venues_test_user:
					pred_rating = self.recommender.predict_score_by_ratings(venue, user_training_venues)
					actual_rating = user_data[user_data['venue_id'] == venue].iloc[0]['rating']
					
					
					if (pred_rating > 0 and actual_rating>0):
						error_mae = abs(pred_rating - actual_rating)
						error_rmse = pow(pred_rating - actual_rating, 2)
						rmse += error_rmse
						sum_mae +=error_mae
						n_venues +=1
						n_venues_rmse +=1
				
				#if for all the venues in test there is prediction equal to zero or the actual rating is zero then the error is 0   
				if(n_venues>0):
					mae += sum_mae/n_venues

		root_mean_squared_error = sqrt(rmse/n_venues_rmse)
		mean_average_error = mae/number_of_users
		return mean_average_error, root_mean_squared_error
