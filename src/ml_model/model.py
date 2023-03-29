# %% [markdown]
# # Later we'll customize this more

# %%
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

# %% [markdown]
# <a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%202.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# Run this to ensure TensorFlow 2.x is used
# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass

# %%
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# from data import reviews

# %%
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000


# # %%
# !wget --no-check-certificate \
#     https://storage.googleapis.com/learning-datasets/sarcasm.json \
#     -O /tmp/sarcasm.json


# %%
# with open("/tmp/sarcasm.json", 'r') as f:
with open("/data/reviews.json", 'r') as f:
    datastore = json.load(f)

# datastore = reviews
sentences = []
labels = []

for item in datastore:
    print(item)
    sentences.append(item['Review'])
    labels.append(item['Liked'])

# %%
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# %%
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# %%
# Need this block to get it to work with TensorFlow 2.x
import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# %%
model.summary()


# %%
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# validation_data: Data on which to evaluate
#                 the loss and any model metrics at the end of each epoch.
#                 The model will not be trained on this data. Thus, note the fact
#                 that the validation loss of data provided using
#                 `validation_split` or `validation_data` is not affected by
#                 regularization layers like noise and dropout.
#                 `validation_data` will override `validation_split`.
#                 `validation_data` could be:
#                   - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
#                   - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
#                     arrays.
#                   - A `tf.data.Dataset`.

# %%
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    # plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, f'val_{string}'])
    plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# %%
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# print(decode_sentence(training_padded[0]))
# print(training_sentences[2])
# print(labels[2])

# %%
e = model.layers[0]
weights = e.get_weights()[0]
# print(weights.shape) # shape: (vocab_size, embedding_dim)

# %%
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

# %% Testing model predictions
# sentence = ["The food is perfect, but is horrible."]
# sequences = tokenizer.texts_to_sequences(sentence)
# padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(model.predict(padded))


# # %%
# sentence = ["The food is horrible."]
# sequences = tokenizer.texts_to_sequences(sentence)
# padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# print(model.predict(padded))

# %%
from statistics import mean

# %%
def eval_reviews(reviews):
    padded = None
    for r in reviews:
      sequences = tokenizer.texts_to_sequences(r)
      padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    return mean((model.predict(score) for score in padded))


