# NLP-Twitter-Ratings

This project is a contribution to the seminar "Applied Deep Learning for NLP" held at TUM during the summer term of 2021. 

### What is it about?
It is about predicting the "rating" of a given tweet inside a specified user's "bubble".  
The "rating" is defined as a custom metric consisting of the ratios `#Likes/#Followers`, `#Retweets/#Followers` and so on.
The reason behind this is that absolute numbers like the amount of comments or retweets are largely dependent on the size of the user's profile on Twitter. Also, because likes, retweets, comments, ... follow extremely positive-skewed log-normal distributions and the necessity of transforming such distributions (see "Problems" section), a custom metric has the advantage that it doesn't need to be transformed back into its original space.

A "bubble" is a user's "ego-net", i.e. the users who stand in some kind of relationship with the user, with a degree of not larger than 3. This ensures that the used language inside the bubble hopefully reflects the one the user tends to use and that common topics or language patterns can be learned by a machine learning model.  

This project assumes and wants to show or at least try to show that it is possible to predict the mentioned "rating" metric in some manner due to the fact that Twitter users tend to form those "bubbles" which are separate communities of similar interests, political views or other defining aspects.

## Structure of the Project
Two separate Python environments are needed due to some incompatibilities with versions of `numpy`. The requirements for those `conda` and `pip` environments can be found in the provided files.  

The project is decomposed in a few Jupyter notebooks which need to be run in a specific sequence:

1. `twitter_miner.ipynb`: This notebook is used to gather a dataset for a given Twitter user. It needs access to Twitter's API. A list of user ids gained through tools like `nucoll` and `Gephi` can be specified.  

2. `embedding.ipynb`: Here, a word2vec word embedding using `gensim` is trained on the previously downloaded data. It is used later on in machine learning models. Due to incompatibilities between `numpy` and `gensim`, this notebook needs a different environment than the others.  

3. `classifier.ipynb`: This notebook is used to train a binary classifier in order to predict whether a tweet may be considered "good" or "bad". The classification is necessary due to the multimodal nature of the transformed ratings (see Problems section).  

4. `regressor.ipynb`: This notebook is used to train two regression models which are than used to predict the tweet's final "rating".  

5. `application.ipynb`: Here, all parts come together. This notebook shows a basic application of this project's models. The project's user can specify a custom string and get a predicted "rating" for this string.  


The directory "results" contains some experiments with different types of models together with their evaluation and some insights into the used data.  


## Results
Our results show that it is indeed possible to predict the "rating" of a given tweet inside a specified bubble - at least to a certain (very small) degree. 
The problems encountered during the experiments show that this is not as trivial as it may sound:

### Problems
The first problem was the distribution of the data (amount of likes, retweets, ...). Due to their log-normal distribution, regression directly on those data failed since it always predicted the used metric's mean.  
This problem was solved by transforming the data via the "Boxcox" procedure so that it approximates a normal distribution as is needed for regression. But transforming the data in such a way led to bimodal distributions with which regression also doesn't seem to be possible. The used regression models always predicted the "center" between both peaks.  
Therefore, the data needed to be "classified", i.e. put into one of two bins, each representing one peak of the bimodal distribution before attempting to run regression on the data.
Two regressors had to be trained - one for each peak. The results seem to show that this helped with the aforementioned problem of the regressor stubbornly predicting either the mean or the "center" between both peaks.  

Since the accuracy, precision and recall on the classifier seem to vary depending on the used dataset and the fact that the tried classifiers seem to perform rather badly, it seems like further gathering of data, improvement of the hyperparameters and perhaps addition of further features (e.g. sentiment analysis) may be needed.  


## Outlook
The project could be improved by tweaking the classifier, which seems to be the most difficult but also most important part of it. As mentioned, this could happen by using better embeddings (even though the `gensim` model seems to perform quite well on its own) and additional features like sentiment analysis.  
Perhaps, the models need to be trained on even larger datasets for way longer than it was possible during the seminar.
