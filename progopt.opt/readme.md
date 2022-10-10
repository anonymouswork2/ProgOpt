This is the repository of *progopt.opt*.

This repository contains several files and folders, they are:

+ ```run.py```: Driver script of *progopt.opt*.

+ ```model-related```: Folder contains all of off-line modules output.

+ ```collect-data.py```: Off-line script collects trainning data.

+ ```mask.py```: Off-line script generates mask.

+ ```model.py```: Off-line script trainning model.

+ ```preprocess-data.py```: Off-line script preprocess data.

+ ```predict-candidate.py```: Script that be invoked by ```run.py```, to predict given program and optimizations.

To run *progopt.opt*, please directly configure ```run.py``` and run with pre-trained model(s).

To reproduce the result, please remove unrelated models in ```run.py```.