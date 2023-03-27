import random, statistics
import ml_model.model as model
from dataclasses import dataclass
import requests

ratings = [random.randint(0, 5) for _ in range(1000)]

ratings = [random.randint(0, 5) for _ in range(1000)]
@dataclass
class Restaurant():
    location: str #temporary - get from Google API
    name: str
    reviews: dict[int]
    ratings: list[int]
    rating_score: int = 0
    review_score: int = 0
    final_score: int = 0

    def get_final_score(self):
        return self.final_score
        
    def set_score(self):
        self.review_score = model.eval_reviews(self.reviews.keys())
        self.rating_score = self.__set_rating_score()
        self.final_score = self.__set_final_score()

    def __set_rating_score(self):
        total = sum(self.ratings)
        norm_ratings = (r/total for r in self.ratings)
        return  statistics.mean(norm_ratings) if self.ratings else 0

    def __set_final_score(self):
        # weight is temporary
        weight = self.__eval_weight()
        return (weight*self.rating_score) + ((1-weight)*self.review_score)
    
    def __eval_weight(self): #may replace with ai weight that Shuyang will do
        return .2
        

class topRecommendations():
    num_recs:int
    restaurants:list[Restaurant()]
    topN: list[Restaurant()]

    def __set_top_N(self, n):
        rs = random.sample(self.num_recs, self.restaurants)
        self.topN = sorted(rs, key = lambda r: r.get_final_score(), reverse=True)[:n]
        

class queryRestaurants():
    num_rest: int
    restaurants:list[Restaurant()]
    query: str

    def __set_restaurants(self):
        """
        make call to Google and get N restaurants
        Query google for restaurants then turn the restaurants
        into Restaurant objects.
        """

        restaurants = []

        for _ in range(self.num_rest):
            rec = self.__query_google(self.query)
            restaurants += [self.__google_to_restaurant(rec)]

        self.restaurants = restaurants

    def __query_google(self) -> dict:
        ...
  
    
    def __google_to_restaurant(self) -> Restaurant:
        ...


    
    
  
    