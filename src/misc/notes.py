# %%
#Generate random restaurant data
import random
import json

restaurants = []
for i in range(500):
    name = f"Restaurant {i}"
    rating = round(random.uniform(0, 1), 1)
    review = round(random.uniform(0, 1), 1)
    restaurants.append({"name": name, "rating": rating, "review": review})

with open("restaurants.json", "w") as f:
    json.dump(restaurants, f, indent=2)

# %%
restaurants


