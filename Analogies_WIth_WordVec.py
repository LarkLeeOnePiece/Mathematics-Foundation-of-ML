import torch
import torchtext
import numpy as np
# download pre-trained word vectors, trained on 6 Billion words, embedding dim 100
vectors = torchtext.vocab.GloVe(name='6B', dim=300, max_vectors=100_000)

# Some other embeddings widely available include:
# (name="42B", dim="300"),
# (name="840B", dim="300"),
# (name="twitter.27B", dim="25"),
# (name="twitter.27B", dim="50"),
# (name="twitter.27B", dim="100"),
# (name="twitter.27B", dim="200"),
# (name="6B", dim="50"),
# (name="6B", dim="100"),
# (name="6B", dim="200"),
# (name="6B", dim="300"),


# Example usage:
u = vectors['bread']    # vector for the word "bread", torch tensor of size (100,)

itos = vectors.itos     # list of the words, mapping indices to words.
stoi = vectors.stoi     # dictionary mapping words to indices.


def most_similar(word, embeddings, top_n=10):
    if word not in embeddings.stoi:
        return []

    # Get the vector for the given word
    word_vector = embeddings.vectors[embeddings.stoi[word]]

    # Calculate cosine similarities with all words
    cosine_similarities = np.dot(embeddings.vectors, word_vector) / (np.linalg.norm(embeddings.vectors, axis=1) * np.linalg.norm(word_vector))

    # Get the indices of top N most similar words
    top_indices = np.argsort(cosine_similarities)[::-1][1:top_n+1]

    # Get the words corresponding to the top indices
    most_similar_words = [embeddings.itos[i] for i in top_indices]

    return most_similar_words

# Test the function
similar_words = most_similar('bread', vectors)
print("10 top similar works wrt bread are: ",similar_words)


def cosine_similarity(u, v):
  return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))



def doesnt_match(words, embeddings):
    if len(words) < 4:
        return "Not enough words to compare."

    # Create a list of word vectors for the given words
    word_vectors = [embeddings.vectors[embeddings.stoi[word]] for word in words]

    # Calculate the cosine similarity matrix (corrected for vector magnitudes)
    similarity_matrix = np.zeros((len(word_vectors), len(word_vectors)))
    for i in range(len(word_vectors)):
        for j in range(len(word_vectors)):
            if i != j:
                similarity = cosine_similarity(word_vectors[i], word_vectors[j])
                similarity_matrix[i][j] = similarity

    # Calculate the average cosine similarity for each word
    avg_similarities = np.mean(similarity_matrix, axis=1)

    # Find the word with the lowest average cosine similarity (least similar word)
    index_of_least_similar = np.argmin(avg_similarities)

    return words[index_of_least_similar]

# Test the function
word_list = ['breakfast', 'cereal', 'dinner', 'lunch']
least_similar_word = doesnt_match(word_list, vectors)
print("The word that doesn't match:", least_similar_word)



# define a function to return the best word that completes an analogy
def analogy(word1, word2, word3):
  # check if the words are in the vocabulary
  if word1 not in vectors.stoi or word2 not in vectors.stoi or word3 not in vectors.stoi:
    return "Sorry, one or more of the words are not in the vocabulary."
  # get the word vectors of the words
  word1_vector = vectors[word1]
  word2_vector = vectors[word2]
  word3_vector = vectors[word3]
  # compute the target vector by adding and subtracting the word vectors
  target_vector = word2_vector - word1_vector + word3_vector
  # initialize a variable to store the maximum similarity and the best word
  max_similarity = -float('inf')
  best_word = None
  # loop through the vocabulary
  for other_word in vectors.stoi:
    # skip the given words themselves
    if other_word in [word1, word2, word3]:
      continue
    # get the word vector of the other word
    other_word_vector = vectors[other_word]
    # compute the cosine similarity between the target vector and the other word vector
    similarity = cosine_similarity(target_vector, other_word_vector)
    # update the maximum similarity and the best word if needed
    if similarity > max_similarity:
      max_similarity = similarity
      best_word = other_word
  # return the best word
  return best_word
# print the result of analogy('man', 'king', 'woman')
print('Analogy complete with: ',analogy('man', 'king', 'woman'))

