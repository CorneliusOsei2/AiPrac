from classes.restaurant import Restaurant
from parse_json import data


restaurants: list[Restaurant] = []
for name, reviews, ratings in data:
    restaurant = Restaurant(name=name, reviews=reviews, ratings=ratings)
    restaurant.set_score()
    restaurants.append(restaurant)

for restaurant in restaurants:
    restaurant.set_review_score(reviews)
