
import numpy as np
import time
import torch
from torch import nn
"""
  1   2    3
1 1   0    1   	     matchers en rekening houden met 
  
2 0   1    1  

3 1   1    1



[ 1: 0.6, 2: 0.2, 3: 0,4 ]

"""

def tel_buren_op(zaal_logits):
    threshold = 1/(len(rechtstreekse_verbindingen) * 5)
    # threshold = round(threshold, 2)
    # threshold += 0.01
    added_buren = set()
    kansen = m(torch.from_numpy(zaal_logits))
    for i in range(len(zaal_logits)):
        el = zaal_logits[i]
        if kansen[i] > threshold:
            # 1 hoger
            if i+1 < len(zaal_logits) and kansen[i+1] < threshold:
                added_buren.add(i+1)
                zaal_logits[i+1] += 0.01
            
            if i-1 < len(zaal_logits) and kansen[i-1] < threshold :
                added_buren.add(i-1)
                zaal_logits[i-1] += 0.01
    return zaal_logits
            

rechtstreekse_verbindingen = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])

zaal_logits = np.empty(len(rechtstreekse_verbindingen) * 5)
zaal_logits.fill(1/(len(rechtstreekse_verbindingen) * 5))

threshold = 1/(len(rechtstreekse_verbindingen) * 5)
threshold -= 0.01
print("threshold", threshold)

    
# Probeer te matchen met alles
# Verhoog de probabiliteit van de top 3 met een genormaliseerde matching score --> de rest wordt allemaal verlaagd

# Per seconde in de video wordt de probabiliteit van de aanliggende van de hoogste probabiliteiten groter

# import ipdb; ipdb.set_trace()
print(zaal_logits[0])
zaal_logits[0] = 2


    
    
while True:
    # softmax of zaal_logits with torch
    m = nn.Softmax(dim=0)
    zaal_logits = tel_buren_op(zaal_logits)
    print(m(torch.from_numpy(zaal_logits)))
    # tel bij alle een waarde op
    # print(zaal_logits)
    time.sleep(0.01)
    