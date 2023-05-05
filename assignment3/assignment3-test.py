
import numpy as np
import time
import torch
from torch import nn
from scipy.stats import entropy
from copy import deepcopy


"""
  1   2    3
1 1   0    1   	     matchers en rekening houden met 
  
2 0   1    1  

3 1   1    1



[ 1: 0.6, 2: 0.2, 3: 0,4 ]

"""

def get_buren():
    pass

def tel_buren_op(zaal_logits):
    threshold = 1/(len(rechtstreekse_verbindingen) * 5)
    added_buren = set()
    kansen = m(torch.from_numpy(zaal_logits))
    for i in range(len(zaal_logits)):
        if kansen[i] > threshold:
            # 1 hoger
            zaal = 1
            indices = get_buren(zaal)
            if i+1 < len(zaal_logits) and kansen[i+1] < threshold:
                added_buren.add(i+1)
                zaal_logits[i+1] += 0.01
            
            if i-1 < len(zaal_logits) and kansen[i-1] < threshold :
                added_buren.add(i-1)
                zaal_logits[i-1] += 0.01
    return zaal_logits
            

if __name__ == "__main__":

    rechtstreekse_verbindingen = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])

    zaal_logits = np.empty(len(rechtstreekse_verbindingen) * 5)
    zaal_logits.fill(1/(len(rechtstreekse_verbindingen) * 5))
    target = deepcopy(zaal_logits)
    zaal_logits[0] = 2

        
    # Probeer te matchen met alles
    # Verhoog de probabiliteit van de top 3 met een genormaliseerde matching score --> de rest wordt allemaal verlaagd
    # Per seconde in de video wordt de probabiliteit van de aanliggende van de hoogste probabiliteiten groter

    tolerance = 1e-6

    # Initialize the KL divergence to a large value
    kl_div = np.inf

    m = nn.Softmax(dim=0)
    

    while kl_div > tolerance:
        # softmax of zaal_logits with torch
        zaal_logits = tel_buren_op(zaal_logits)
        zaal_normalized = m(torch.from_numpy(zaal_logits))
        # tel bij alle een waarde op
        # print(zaal_logits)
        kl_div = entropy(zaal_normalized, target)
        # print("zaal_normalized: " , zaal_normalized)
        # print("target: ", target)
        print(zaal_normalized)
    