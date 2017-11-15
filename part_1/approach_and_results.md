# Approach

I analyzed the Jester dataset (http://eigentaste.berkeley.edu/dataset/) which compiles millions of ratings of jokes by anonymous users. Demos of jokes can be found here: http://eigentaste.berkeley.edu/.

For feasibility of computation, I worked with a reduced datset comprising 10,000 users and their existing ratings on 100 jokes. As expected, not all 100 jokes were rated by every user.

Using the surprise Python scikit package, I built a model-based matrix factorization algorithm using SVD, and I built a neighborhood-based algorithm using KNN.

I built a training and testing set in order to try to predict ratings the model has not seen before and compare its prediction to the ground truth rating.

I evaluated each model on a range of hyperparameters as well as a range of dataset sizes. The results follow. I give RMSE and MAE error rates which the models yielded.



# Results

__SVD hyperparameter optimization:__

4 different hyperparameters were tuned with the following values:
{'n_factors': [20, 100], 'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}

This setup yields a combination of 16 different parameter combinations. These are too many to list here and cannot be plotted due to there being more than 3 dimensions, but the list can be seen if command #4 is run in part_1_notebook_svd.

DATASET SIZE: 10,000
The best set of parameters found was
Parameters combination 12 of 16
params:  {'lr_all': 0.005, 'reg_all': 0.4, 'n_factors': 100, 'n_epochs': 10}
------------
Mean RMSE: 3.7164
Mean MAE : 2.7140

DATASET SIZE: 5,000
Best set of parameters found was
Parameters combination 12 of 16
params:  {'lr_all': 0.005, 'reg_all': 0.4, 'n_factors': 100, 'n_epochs': 10}
------------
Mean RMSE: 3.7323
Mean MAE : 2.6961

DATASET SIZE: 1,000
Best set of parameters found was
Parameters combination 12 of 16
params:  {'lr_all': 0.005, 'reg_all': 0.4, 'n_factors': 100, 'n_epochs': 10}
------------
Mean RMSE: 3.9817
Mean MAE : 2.6880

DATASET SIZE: 100
Best set of parameters found was
Parameters combination 12 of 16
params:  {'lr_all': 0.005, 'reg_all': 0.4, 'n_factors': 100, 'n_epochs': 10}
------------
Mean RMSE: 6.3159
Mean MAE : 3.2551

__KNN hyperparameter optimization:__
Plot of 10,000 user evaluation


![/Results](https://i.imgur.com/tezRhzu.png)

DATASET SIZE: 10,000
Best performance was
k = 40
RMSE: 4.0436
MAE:  2.9098 (plotted above)

DATASET SIZE: 5,000
Best performance was
k = 40
RMSE: 4.1398
MAE:  2.9507

DATASET SIZE: 1,000
Best performance was
k = 20
RMSE: 4.4018
MAE:  2.9877

DATASET SIZE: 100
Best performance was
k = 20
RMSE: 7.5166
MAE:  3.7075


# Discussion

Hyperparamter optimization:
For SVD, a higher number of epochs was naturally better for performance (more iterations of gradient descent to descend to the minimum). A higher number of factors, interestingly, led to better performance. Perhaps this means that there are many dimensions that describe the jokes that appear. The higher learning rate beat the smaller learning rate, which suggests the smaller learning rate was affecting a movement speed that was too slow in the algorithm. Finally the smaller regularization term yielded better performance. Regularization is used to reduce model complexity and in this case it was not needed.

For KNN, a k of 40 was optimal. This is purely due to the size of the dataset. Observe that when only 1,000 or 100 users comprise the dataset, a k of 20 was more optimal.

In future design, I may attempt to tune KNN hyperparameters further. KNN takes a long time to run and it is difficult to get results for all combinations of parameter settings. In the above evaluations, cosine similarity and and item-based recommendation were used. In future design, I may attempt a pearson similarity function and a user-based recommendation scheme.

The solutions and results have been fairly successful with a MAE of roughly 2.7 in the SVD algorithm. I am confident putting these results to use in a recommender system but I believe there is more to improve to bring the error rate down. More data, more testing, and more hyperparamter optimization is the way to bring the error down further.
