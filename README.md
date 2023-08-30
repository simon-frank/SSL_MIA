# SSL_MIA

Self-Supervised Learning for Medical Image Analysis

In the following a quick overview over the content of the files:

- train.py: pretraining the model with the dataset specificied in the config using the Barlow Twin Method
- evaluation.py: evaluating the pre-trained features
- finetuning_evaluation.py: calculate the accuracies on the test set and create confusion matrix
- finetuning.py: finetuning the pretrained model on the downstream dataset
- utils: different kind of util files
-  
