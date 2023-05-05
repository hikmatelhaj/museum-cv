import numpy as np

# Define the set of states
states = ['zaal1', 'zaal2']

# Define the emission probabilities
emission_probs = {
    'zaal1': {'event1': 0.2, 'event2': 0.6},
    'zaal2': {'event1': 0.8, 'event2': 0.4}
}

# Define the transition probabilities
transition_probs = {
    'zaal1': {'zaal1': 0.7, 'zaal2': 0.3},
    'zaal2': {'zaal1': 0.4, 'zaal2': 0.6}
}

# Define the initial probabilities
initial_probs = {'zaal1': 0.5, 'zaal2': 0.5}

# Define the input sequence
sequence = ['event1', 'event2']

# Implement the Viterbi algorithm
delta = np.zeros((len(sequence), len(states)))
psi = np.zeros((len(sequence), len(states)), dtype=int)

for i in range(len(states)):
    delta[0][i] = initial_probs[states[i]] * emission_probs[states[i]][sequence[0]]

for t in range(1, len(sequence)):
    for j in range(len(states)):
        delta[t][j] = np.max([delta[t-1][i] * transition_probs[states[i]][states[j]] * emission_probs[states[j]][sequence[t]] for i in range(len(states))])
        psi[t][j] = np.argmax([delta[t-1][i] * transition_probs[states[i]][states[j]] for i in range(len(states))])

path = np.zeros(len(sequence), dtype=int)
path[-1] = np.argmax(delta[-1])
for t in range(len(sequence)-2, -1, -1):
    path[t] = psi[t+1][path[t+1]]

print('Most likely sequence of states:', [states[i] for i in path])