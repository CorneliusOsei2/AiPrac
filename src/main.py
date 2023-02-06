import json
import os

from classes.restaurant import Restaurant, train, Recommendations
from query import main

restaurants = []


def init_restaurants(n):
    """
    make a random Restaurant to use in testing the set_scores function in restaurant.py
    """
    global restaurants

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    restaurant_file_path = os.path.join(root_dir, "src", "data", "all_restaurants.json")

    with open(restaurant_file_path, "r") as f:
        restaurant_data = json.load(f)

    i = 0
    for k, v in restaurant_data.get("data").items():
        if i == n:
            break
        i += 1
        rest = Restaurant(name=k)
        for item in v["reviews_and_ratings"]:
            rev, rat = item["Review"], item["Rating"]
            rest.reviews.append(rev)
            rest.ratings.append(rat)
        restaurants.append(rest)


import io
import sys


def make_restaurants(n=100, display=False):
    global restaurants

    init_restaurants(n)

    sys.stdout = io.StringIO()
    train()
    [r.set_scores() for r in restaurants]
    sys.stdout = sys.__stdout__

    if display:
        for r in restaurants:
            print(f"SCORES FOR {r.name}\n")
            print(
                f"\nreview score is {r.review_score}\n rating score is {r.rating_score}\n final score is {r.final_score}"
            )
            print("\n----------------------------\n")


def make_recommendations(n=5):
    global restaurants

    recs = Recommendations(restaurants=restaurants)
    recs.set_top_N(n)

    for i in range(len(recs.best)):
        r = recs.best[i]
        print(f"{i+1}. {r.name}\n")
        print(f"final score is {r.final_score}")
        print("\n----------------------------\n")


if __name__ == "__main__":
    main()
    make_restaurants()
    make_recommendations()
