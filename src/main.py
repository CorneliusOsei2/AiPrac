from classes.restaurant import Restaurant
from utils import parse_json

restaurants = []
def get_restaurant_data():
    response = ...
    return parse_json(response)

def create_restaurants(restaurants):
    for rest in restaurants:
        restaurant = Restaurant(name=rest.name, reviews=rest.reviews, ratings=rest.ratings)
        restaurant.set_score()
        restaurants.append(restaurant)
