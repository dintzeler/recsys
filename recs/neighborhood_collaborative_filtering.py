from recommender.models import Ratings, Similarity

from django.db.models import Q
import time
from datetime import datetime
from decimal import Decimal

class NeighborhoodBasedRecs(object):

	def __init__(self, neighborhood_size=15, min_sim=0.0):
		self.neighborhood_size = neighborhood_size
		self.min_sim = min_sim
		self.max_candidates = 400

	def recommend_items(self, user_id, num=6):

		active_user_items = Ratings.objects.filter(user_id=user_id)
		return self.recommend_items_by_ratings(user_id, active_user_items.values())

	def recommend_items_by_ratings(self, user_id, active_user_items ,num = 6):
		
		#load the ratings of user_id
		
		
		if len(active_user_items) == 0:
			return {}

		venues = {venue['venue_id']: venue['rating'] for venue in active_user_items}

		user_mean = sum(venues.values()) / len(venues)



		#candidate items are the ones that have at least one similarity with the venues user_id has rated
		candidate_items = Similarity.objects.filter(Q(source__in=venues.keys())&~Q(target__in=venues.keys()))

		#we keep the first max_candidates items with the biggest similarity
		candidate_items = candidate_items.order_by('-similarity')

		recs = dict()

		for candidate in candidate_items:
			target = candidate.target
		
			pre = 0
			sim_sum = 0

			#save in list rated_items all the similarities of candidate_item
			rated_items = [i for i in candidate_items if i.target == target][:self.neighborhood_size]

			if len(rated_items) > 1:

				#calculate predicted rating
				for sim_item in rated_items:
#					r = Decimal(venues[sim_item.source] - user_mean)
					r = Decimal(venues[sim_item.source])
					pre += sim_item.similarity * r
					sim_sum += sim_item.similarity
				if sim_sum > 0:
					#save prediction and also the similar items to target
					#recs[target] = {'prediction': Decimal(user_mean) + pre / sim_sum,'sim_items': [r.source for r in rated_items]}
					recs[target] = {'prediction': pre / sim_sum,'sim_items': [r.source for r in rated_items]}
		#show topN items
		sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['prediction']))[:num]
		
		return sorted_items		

	def predict_score_by_ratings(self, item_id, venue_ids):

		top = Decimal(0.0)
		bottom = Decimal(0.0)
		ids = venue_ids.keys()
		candidate_items = (Similarity.objects.filter(source__in= ids)
											 .exclude(source=item_id)
											 .filter(target=item_id))
		candidate_items = candidate_items.distinct().order_by('-similarity')[:self.max_candidates]

		if len(candidate_items) == 0:
			return 0

		for sim_item in candidate_items:
			r = venue_ids[sim_item.source]
			top += sim_item.similarity * r
			bottom += sim_item.similarity

		return Decimal(top/bottom)