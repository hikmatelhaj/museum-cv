# computervision-2023
Project Computervision

# Assignment 1: painting detection


# Assignment 2: matching
Files located in the folder `assignment2`. The files `histogram.pkl`, `keypoints.pkl` and `descriptors.pkl` are mandatory for the code to run. These files can also be created with the function `create_keypoints_and_color_hist_db()`

# Assignment 3: localization
Files located in the folder `assignment3`. The file `label_video.py` is used to manually label the videos. The results are saved in `labels`.



### Transition matrix

The file ```assignment3/transition_matrix.py``` contains the code for creating the transition matrix. 

The indices used in the matrix correspond to the following plan: 



![adapted_plan_msk](https://github.ugent.be/storage/user/10152/files/2b632b56-6c3e-4557-8797-7d43bfdf086e)

### Hidden Markov Model

The code found in ```assignment3/assignment3.py``` includes the functionality to determine the room probabilities and localize the user based on a video in the museum.

The function ```calculate_hmm``` uses the ```CategoricalHMM``` model from hmmlearn, a Python library for hidden markov models, to determine the room probabilities.

The file ```assignment3/geomap.py``` together with ```assignment3/coords.csv``` provide a visualization of the room probabilities in the form of a heatmap. This is used in ```assignment3/assignment3.py``` to show a clear image of the probabilities when executing ```calculate_hmm```.

