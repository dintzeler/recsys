from decimal import Decimal

from recommender.models import Ratings

from django.db.models import Count
from django.db.models import Q
from django.db.models import Avg

class BaselineRecommender(object):

    def __init__(self):
        self.sorted_items = self.load_sorted_items()

    def recommend_items(self, user_id, num=6):

        active_user_items = Ratings.objects.filter(user_id=user_id)
        return self.recommend_items_by_ratings(user_id, active_user_items.values())

    def predict_score_by_ratings(self, item_id, venue_ids):
        avg_rating = Ratings.objects.filter(venue_id=item_id).values('venue_id').aggregate(Avg('rating'))
        return avg_rating['rating__avg']




    def recommend_items_by_ratings(self, user_id, active_user_items, num=6):
        item_ids = [i['venue_id'] for i in active_user_items]
        for item in self.sorted_items:
            if item[0] in item_ids:
                self.sorted_items.remove(item)

        return(self.sorted_items[:num])


    def load_sorted_items(self):
        pop_items = Ratings.objects.values('venue_id').annotate(Count('user_id'),Avg('rating'))
        recs = {i['venue_id']: {'prediction': i['rating__avg'], 'pop': i['user_id__count']} for i in pop_items}
        sorted_items = sorted(recs.items(), key=lambda item: -float(item[1]['pop']))[:500]
        return sorted_items

