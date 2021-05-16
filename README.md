## Natural language processing using Multinomial Naive Bayes and Logistic Regression
![](https://img.shields.io/badge/-python-informational?style=for-the-badge&logo=python&logoColor=white&color=3776AB)

Study conducted as a part of McGill COMP551 (Applied Machine Learning). <br/>
The full paper may be read [here](https://github.com/Unixado/NLP_Text_Classification-MNB-LR/blob/master/Report/Report.pdf).<br/>
Further results can be found in the [Notebook](https://github.com/Unixado/NLP_Text_Classification-MNB-LR/blob/master/src/COMP551_Project2.ipynb).


### Abstract
In this project, we studied the performance of **linear classification models, namely multinomial naive Bayes and logistic regression on two textual datasets for binary and multiclass classification**. Although the logistic regression model outper-formed multinomial naive Bayes on the **IMDb dataset** in terms of accuracy by 5 (87% vs 82%), both models achieved similar accuracies when applied on the **20newsgroup** dataset with about 70%. Notwithstanding, it is worth mentioning that the naive Bayes model is computationally more expensive than the logistic regression model, making logistic regression faster and more energy-efficient classifier. We used 5-fold cross-validation to tune the hyperparameters of our models on both datasets to achieve the best performance. We compared the models implemented from scratch to Scikit learn models where applicable. For instance, our logistic regression model showed better accuracy for binary classification on the IMDb
dataset than the Scikit-learn model.
Furthermore, we explored the impact of using subsets of training data and features on the accuracy of our models. We
observed only slight changes in the models’ accuracy as a smaller set of features was utilised. However, while decreasing the
number of features improved the models’ performance on the IMDb dataset, it somewhat decreased the performance on the
20newsgroup dataset. We found that the performance of the models on the 20newsgroup dataset depended more heavily
on the amount of information they were trained on compared to the IMDb dataset. For instance, while enlarging the size
of data the models were trained on increased the accuracy on both datasets, this gain in accuracy was more significant on
the 20newgroup dataset. These findings are consistent with the results in [1]. To further evaluate our models, performance
metrics such as the ROC curve and confusion matrix were studied.
