from classes.restaurant import Restaurant
from utils import parse_json
import random


#### Dummy data
import json
import os

# Get the absolute path of the project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute path of the reviews.json file
reviews_file_path = os.path.join(root_dir, "src", "data", "reviews.json")

with open(reviews_file_path, "r") as f:
    review_data = json.load(f)


restaurants = []
def get_restaurant_data():
    response = ...
    return parse_json(response)

def create_restaurants(restaurants):
    for rest in restaurants:
        restaurant = Restaurant(name=rest.name, reviews=rest.reviews, ratings=rest.ratings)
        restaurant.set_score()
        restaurants.append(restaurant)

def make_dummy_restaurant(n):
    reviews = [random.choice(review_data)["Review"] for _ in range(n)]
    ratings = [random.uniform(0,5) for _ in range(n)]

    return Restaurant(
        name="test",
        reviews=reviews,
        ratings=ratings,
        rating_score=0,
        review_score=0,
        final_score=0,
    )

if __name__ == "__main__":
    restaurant = make_dummy_restaurant(400)

    restaurant.set_scores()
    rev, rat, final = restaurant.get_scores()

    print(f"review score is {rev}\n rating score is {rat}\n final score is {final}")
    # main()