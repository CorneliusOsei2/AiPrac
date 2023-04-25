import random, statistics
from ml_model.model import eval_reviews, eval_weights
from dataclasses import dataclass

from typing import List


# ratings = [random.randint(0, 5) for _ in range(1000)]
@dataclass
class Restaurant:
    # location: str #temporary - get from Google API
    name: str
    reviews: List[str] = []
    ratings: List[int] = []
    rating_score: int = 0
    review_score: int = 0
    final_score: int = 0

    def get_scores(self):
        """
        return a dict containing all the scores
        """
        return {
            "review_score": self.review_score,
            "rating_score": self.rating_score,
            "final_score": self.final_score,
        }

    def set_scores(self):
        """
        update all the scores of a restaurant
        """
        self.__set_review_score()
        self.__set_rating_score()
        self.__set_final_score()

    def __set_review_score(self):
        """
        update review_score as the score calculated from eval_reviews()
        """
        self.review_score = eval_reviews(self.reviews)

    def __set_rating_score(self):
        """
        set rating_score to a score
        """
        total = sum(self.ratings)
        norm_ratings = (r / total for r in self.ratings)
        self.rating_score = statistics.mean(norm_ratings) if self.ratings else 0

    def __set_final_score(self):
        # weight is temporary
        rating_weight, review_weight = eval_weights(self.ratings, self.reviews)
        self.final_score = (rating_weight * self.rating_score) + (
            review_weight * self.review_score
        )


@dataclass
class Recommendations:
    num_recs: int
    restaurants: "list[Restaurant()]"
    topN: "list[Restaurant()]"

    def __set_top_N(self, n):
        """
        sort and get the top N restaurants based on their final_score
        """
        rs = random.sample(self.num_recs, self.restaurants)
        self.topN = sorted(
            rs, key=lambda r: r.get_scores()["final_score"], reverse=True
        )[:n]
