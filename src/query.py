from __future__ import print_function
from collections import defaultdict
import json
import requests

"""
Command:
python query.py --term="restaurant" --location="Ithaca, NY"`
"""

import argparse
from urllib.parse import quote

API_KEY = "9J3O7n5PJAs9mUCbg_--IyC3DJdaLAa8C_Pi5cvABnphQDV-hqccCZRpD5spZxSF6mnrZ71RW1EQaVLFDipIBSFQujVPSALW8tjEhX-NNBmhxK2MaLpDttAVxtskZHYx"
API_HOST = "https://api.yelp.com"
SEARCH_PATH = "/v3/businesses/search"
BUSINESS_PATH = "/v3/businesses/"


DEFAULT_TERM = "restaurant"
DEFAULT_LOCATION = "Ithaca, NY"


def get_data(host, path, url_params={}):
    url = f"{host}{quote(path.encode('utf8'))}"
    headers = {
        "Authorization": "Bearer %s" % API_KEY,
    }

    print(f"Getting available restaurants from {url} ..../.../...")
    response = requests.request("GET", url, headers=headers, params=url_params)
    return response.json()


def search(search_term, location):
    url_params = {
        "term": search_term.replace(" ", "+"),
        "location": location.replace(" ", "+"),
    }
    return get_data(API_HOST, SEARCH_PATH, url_params=url_params)


def get_restaurant(restaurant_id):
    business_path = BUSINESS_PATH + restaurant_id
    return get_data(API_HOST, business_path)


def query_api(search_term, location):
    response = search(search_term, location)
    restaurants = response.get("businesses")

    if not restaurants:
        print(f"Sorry, we could not find any restaurants in {location}.")
        return

    restaurants_reviews = defaultdict(dict)
    for restaurant in restaurants:
        reviews = get_restaurant(restaurant["id"] + "/reviews")["reviews"]

        for review in reviews:
            if not restaurant["name"] in restaurants_reviews["data"]:
                restaurants_reviews["data"][restaurant["name"]] = [
                    {"Review": review["text"], "Rating": review["rating"]}
                ]
            else:
                restaurants_reviews["data"][restaurant["name"]].append(
                    {"Review": review["text"], "Rating": review["rating"]}
                )

    return write_to_json(restaurants_reviews)


def write_to_json(restaurants_reviews):
    json_object = json.dumps(restaurants_reviews, indent=4)

    with open("restaurants_reviews_ratings.json", "w") as outfile:
        outfile.write(json_object)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-q",
        "--term",
        dest="term",
        default=DEFAULT_TERM,
        type=str,
        help="Search term (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--location",
        dest="location",
        default=DEFAULT_LOCATION,
        type=str,
        help="Search location (default: %(default)s)",
    )

    input_values = parser.parse_args()
    query_api(input_values.term, input_values.location)


if __name__ == "__main__":
    main()
