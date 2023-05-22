import os, json

dirs = [ f.path for f in os.scandir("labels") if f.is_dir() ]

no_matches = 0
bins_FP = {}
bins_TP = {}
bins_FP[0.0] = 0
bins_TP[0.0] = 0
bins_FP[0.1] = 0
bins_TP[0.1] = 0
bins_FP[0.2] = 0
bins_TP[0.2] = 0
bins_FP[0.3] = 0
bins_TP[0.3] = 0
bins_FP[0.4] = 0
bins_TP[0.4] = 0
bins_FP[0.5] = 0
bins_TP[0.5] = 0
bins_FP[0.6] = 0
bins_TP[0.6] = 0
bins_FP[0.7] = 0
bins_TP[0.7] = 0
bins_FP[0.8] = 0
bins_TP[0.8] = 0
bins_FP[0.9] = 0
bins_TP[0.9] = 0
for dir in dirs:

    filename_TP = f'{dir}/{os.path.basename(dir)}_TP.json'
    filename_FP = f'{dir}/{os.path.basename(dir)}_FP.json'
    filename_no_match = f'{dir}/{os.path.basename(dir)}_no_match.txt'
    with open(filename_TP, 'r') as file:
        score_TP = json.load(file)
        
    with open(filename_FP, 'r') as file:
        score_FP = json.load(file)
        
    with open(filename_no_match, "r") as file:
        no_matches += int(file.readline())
    
    for key, value in score_TP.items():
        bins_TP[float(key)] += int(value)
    
    for key, value in score_FP.items():
        bins_FP[float(key)] += int(value)
        

print(bins_FP)
print(bins_TP)
print(no_matches)

for key, value in bins_FP.items():
    if value == 0:
        bins_FP[key] = 1
        
for key, value in bins_TP.items():
    if value == 0:
        bins_TP[key] = 1

filename_TP = f'labels/final_TP.json'
filename_FP = f'labels/final_FP.json'
filename_no_match = f'labels/final_no_match.txt'
with open(filename_TP, 'w') as file:
    json.dump(bins_TP, file)
    
with open(filename_FP, 'w') as file:
    json.dump(bins_FP, file)

with open(filename_no_match, "w") as file:
    file.write(str(no_matches))