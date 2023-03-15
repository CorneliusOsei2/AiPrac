import random, statistics, math
from dataclasses import dataclass
ratings = [random.randint(0, 5) for _ in range(1000)]




@dataclass
class Restaurant():
    location: str #temporary - get from Google API
    name: str
    reviews: list[str]
    ratings: list[int]
    rating_score: int = 0
    review_score: int = 0
    final_score: int = 0

    def get_final_score(self):
        return self.final_score
        
    def set_score(self):
        self.review_score = self.__set_review_score()
        self.rating_score = self.__set_rating_score()
        self.final_score = self.__set_final_score()
    
    def __set_review_score(self):
        ...
        return statistics.mean(self.eval_review(r) for r in self.reviews) if self.reviews else 0
        
    def __set_rating_score(self):
        return  statistics.mean(self.ratings) if self.ratings else 0

    def __set_final_score(self):
        # * temporary
        return (self.rating_score + self.review_score)/2
        ...
        
data = [{"name": ..., "reviews": ..., "ratings": ...}]

restaurants: list[Restaurant] = []
for name, reviews, ratings in data:
    restaurant = Restaurant(name=name, reviews=reviews, ratings=ratings)
    restaurant.set_score()
    restaurants.append(restaurant)

for restaurant in restaurants:
    restaurant.set_review_score(reviews)


class topRecommendations():
    num_recs:int
    restaurants:list[Restaurant()]
    topN: list[Restaurant()]

    def __set_top_N(self):
        rs = random.sample(self.num_recs, self.restaurants)
        self.topN = sorted(rs, key = lambda r: r.get_final_score(), reverse=True)
        

class queryRestaurants():
    num_rest: int
    restaurants:list[Restaurant()]
    query: str

    def __set_restaurants(self):
        """
        make call to Google and get N restaurants
        """

        restaurants = []

        for _ in range(self.num_rest):
            rec = self.__query_google(self.query)
            restaurants += [self.__google_to_restaurant(rec)]

        self.restaurants = restaurants

    def __query_google(self):
        ...
    
    def __google_to_restaurant(self):
        ...


    
    
  
    