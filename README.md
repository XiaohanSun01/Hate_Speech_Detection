# Tweet Hate Speech Detection

Detect the performance of Tweet hate speech detection task on five machine learning models with the cross-dataset evaluation.  

## Description

In recent years, social media platforms have significantly increased, allowing people to communicate and share information worldwide in real time. While social media platforms offer many benefits, there are also several disadvantages, especially the prevalence of hate and provocative speech. As the ninth most popular social media platform, Twitter demonstrates its popularity among diverse populations. In this project, I explored the "Tweet Hate Speech Detection" task, starting by collecting different available online datasets with the help of Twitter API. Then, after processing data cleaning and visualizations, the study focuses on automatically identifying hate speech in tweets by training several machine learning models on one dataset and testing on others. It provides a comprehensive overview of tweet hate speech detection, exploring its challenges, strengths, limitations, and future research directions.


## Getting Started

### Dependencies

* Python, Jupyter Notebook

### Executing program

* Download the whole project via GitHub.
* To run the file "Data_Collection_and_Cleaning.ipynb" successfully, you need to create a "config.py" file. Below is what the file looks like, you should copy and paste your associated and generated keys and tokens from the Twitter API v2. 
```
keys = dict(
    api_key = "Your api_key",
    api_secret = "Your api_secret",
    access_token = "Your access_token", 
    token_secret = "Your token_secret",
    bearer_token = "Your bearer_token"
)

```

## Authors

* Xiaohan Sun
* sunxiaohan0401@gmail.com


## Acknowledgments

* [Functions to Obtain Tweets from API](https://github.com/datascisteven/Medium-Blogs/tree/main/Hate-Tweet-Detector)

