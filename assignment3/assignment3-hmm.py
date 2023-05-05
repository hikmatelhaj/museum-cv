# import numpy as np

# # Define the set of states
# states = ['zaal1', 'zaal2']

# # Define the emission probabilities
# emission_probs = {
#     'zaal1': {'event1': 0.2, 'event2': 0.8},
#     'zaal2': {'event1': 0.6, 'event2': 0.4}
# }

# # Define the transition probabilities
# transition_probs = {
#     'zaal1': {'zaal1': 0.7, 'zaal2': 0.3},
#     'zaal2': {'zaal1': 0.4, 'zaal2': 0.6}
# }

# # Define the initial probabilities
# initial_probs = {'zaal1': 0.5, 'zaal2': 0.5}

# # Define the input sequence
# sequence = ['event1', 'event2']

# # Implement the Viterbi algorithm
# delta = np.zeros((len(sequence), len(states)))
# psi = np.zeros((len(sequence), len(states)), dtype=int)

# for i in range(len(states)):
#     delta[0][i] = initial_probs[states[i]] * emission_probs[states[i]][sequence[0]]

# for t in range(1, len(sequence)):
#     for j in range(len(states)):
#         delta[t][j] = np.max([delta[t-1][i] * transition_probs[states[i]][states[j]] * emission_probs[states[j]][sequence[t]] for i in range(len(states))])
#         psi[t][j] = np.argmax([delta[t-1][i] * transition_probs[states[i]][states[j]] for i in range(len(states))])

# path = np.zeros(len(sequence), dtype=int)
# path[-1] = np.argmax(delta[-1])
# for t in range(len(sequence)-2, -1, -1):
#     path[t] = psi[t+1][path[t+1]]

# print('Most likely sequence of states:', [states[i] for i in path])






import numpy as np
from hmmlearn import hmm

# For this example, we will model the stages of a conversation,
# where each sentence is "generated" with an underlying topic, "cat" or "dog"
states = ["cat", "dog"]
id2topic = dict(zip(range(len(states)), states))
# we are more likely to talk about cats first
start_probs = np.array([0.6, 0.4])

# For each topic, the probability of saying certain words can be modeled by
# a distribution over vocabulary associated with the categories

vocabulary = ["tail", "fetch", "mouse", "food"]
# if the topic is "cat", we are more likely to talk about "mouse"
# if the topic is "dog", we are more likely to talk about "fetch"
emission_probs = np.array([[0.25, 0.1, 0.4, 0.25],
                           [0.2, 0.5, 0.1, 0.2]])

# Also assume it's more likely to stay in a state than transition to the other
trans_mat = np.array([[0.8, 0.2], [0.2, 0.8]])


# Pretend that every sentence we speak only has a total of 5 words,
# i.e. we independently utter a word from the vocabulary 5 times per sentence
# we observe the following bag of words (BoW) for 8 sentences:
observations = [["tail", "mouse", "mouse", "food", "mouse"],
        ["food", "mouse", "mouse", "food", "mouse"],
        ["tail", "mouse", "mouse", "tail", "mouse"],
        ["food", "mouse", "food", "food", "tail"],
        ["tail", "fetch", "mouse", "food", "tail"],
        ["tail", "fetch", "fetch", "food", "fetch"],
        ["fetch", "fetch", "fetch", "food", "tail"],
        ["food", "mouse", "food", "food", "tail"],
        ["tail", "mouse", "mouse", "tail", "mouse"],
        ["fetch", "fetch", "fetch", "fetch", "fetch"]]

# Convert "sentences" to numbers:
vocab2id = dict(zip(vocabulary, range(len(vocabulary))))
def sentence2counts(sentence):
    ans = []
    for word, idx in vocab2id.items():
        count = sentence.count(word)
        ans.append(count)
    return ans

X = []
for sentence in observations:
    row = sentence2counts(sentence)
    X.append(row)

data = np.array(X, dtype=int)

# pretend this is repeated, so we have more data to learn from:
lengths = [len(X)]*5
sequences = np.tile(data, (5,1))


# Set up model:
model = hmm.MultinomialHMM(n_components=len(states),
        n_trials=len(observations[0]),
        n_iter=50,
        init_params='')

model.n_features = len(vocabulary)
model.startprob_ = start_probs
model.transmat_ = trans_mat
model.emissionprob_ = emission_probs
model.fit(sequences, lengths)
logprob, received = model.decode(sequences)

print("Topics discussed:")
print([id2topic[x] for x in received])

print("Learned emission probs:")
print(model.emissionprob_)

print("Learned transition matrix:")
print(model.transmat_)

# Try to reset and refit:
new_model = hmm.MultinomialHMM(n_components=len(states),
        n_trials=len(observations[0]),
        n_iter=50, init_params='ste')

new_model.fit(sequences, lengths)
logprob, received = new_model.decode(sequences)

print("\nNew Model")
print("Topics discussed:")
print([id2topic[x] for x in received])

print("Learned emission probs:")
print(new_model.emissionprob_)

print("Learned transition matrix:")
print(new_model.transmat_)