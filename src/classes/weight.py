# %%
import pandas as pd
from sklearn.decomposition import PCA
from restaurant import Restaurant
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_weight(ratings, reviews):

  # Create a dataframe with the rating and review scores
  X = pd.DataFrame({"ratings":ratings, "reviews":reviews})

  # Fit a PCA model to the data
  pca = PCA().fit(X)

  # Print the explained variance ratios
  # print(pca.explained_variance_ratio_)

  # Determine the weights for rating and review scores
  weights = pca.components_[0]
  rating_weight = weights[0]
  review_weight = weights[1]

  return weights



# # %%
# data


