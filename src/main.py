from classes.restaurant import Restaurant
from utils import parse_json

restaurants = []


def get_restaurant_data():
    """
    get restaurants from a query
    """
    response = ...
    return parse_json(response)


def create_restaurants(restaurants):
    """
    return a list of Restaurant objects made from the data (restaurants).
    Set scores for each
    """
    for rest in restaurants:
        restaurant = Restaurant(
            name=rest.name, reviews=rest.reviews, ratings=rest.ratings
        )
        restaurant.set_score()
        restaurants.append(restaurant)


import json
import os

# # Get the absolute path of the project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Construct the absolute path of the reviews.json file
reviews_file_path = os.path.join(root_dir, "src", "data", "data.json")

with open(reviews_file_path, "r") as f:
    restaurant_data = json.load(f)


def make_restaurants(n):
    """
    make a random Restaurant to use in testing the set_scores function in restaurant.py
    """
    i = 0
    print(restaurant_data.get("data"))
    for k, v in restaurant_data.get("data").items():
        if i == n:
            break
        i += 1
        rest = Restaurant(name=k)
        for rev, rat in v:
            rest.reviews.append(rev)
            rest.ratings.append(rat)

        restaurants.append(rest)


if __name__ == "__main__":
    make_restaurants(10)
    # for r in restaurants:
    #     r.set_scores()
    #     rev, rat, final = r.get_scores()
    #     print(f"review score is {rev}\n rating score is {rat}\n final score is {final}")
