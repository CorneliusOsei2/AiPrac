#  [markdown]
# # Later we'll customize this more

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#  [markdown]
# <a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

import json
import numpy as np
import os
from train import train_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

# Get the absolute path of the project root directory
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the absolute path of the reviews.json file
reviews_file_path = os.path.join(root_dir, "src", "data", "reviews.json")

# Open the reviews.json file
with open(reviews_file_path, "r") as f:
    reviews_data = json.load(f)

def make_model():
    """
    tokenize and split data then return a model trained on that data
    """
    sentences = []
    labels = []

    for item in reviews_data:
        # print(item)
        sentences.append(item['Review'])
        labels.append(item['Liked'])

    training_sentences = sentences[:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[:training_size]
    testing_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    # word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Need this block to get it to work with TensorFlow 2.x
    
    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    model = train_model(vocab_size,embedding_dim,max_length,training_padded, training_labels, testing_padded, testing_labels)

    return model, tokenizer

from statistics import mean
def eval_reviews(reviews):
    """
    Output review score
    """
    model, tokenizer = make_model()
    padded = None
    for r in reviews:
      sequences = tokenizer.texts_to_sequences(r)
      padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    return mean((model.predict(score) for score in padded))

import pandas as pd
from sklearn.decomposition import PCA
def eval_weights(ratings, reviews):
    """
    Output AI adjusted weights for review and rating scores
    """
    # Create a dataframe with the rating and review scores
    X = pd.DataFrame({"ratings":ratings, "reviews":reviews})

    # Fit a PCA model to the data
    pca = PCA().fit(X)

    # Transform the data using the PCA model
    pca.transform(X)

    return pca.components_[0]


