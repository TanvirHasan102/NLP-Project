# NLP-Project

How to run the code? 
Very Simple. I have used only one dataset. Just run the cells. Thats it! All the codes are in "NLP_Project.ipynb" file. 

This report presents a comprehensive analysis of user-generated reviews on Yelp, focusing on sentiment analysis, sentiment prediction, sentiment distribution, attribute extraction, linguistic quality, and machine learning for user rating prediction. The study utilizes various libraries and methods to extract valuable insights from the data. 

#Description of Implementation

Data Loading and Sentiment Calculation: Data is loaded into a Pandas DataFrame. A custom function, calculate_sentiment, uses TextBlob to calculate sentiment scores for each review text.
Sentiment Classification: Another function, classify_sentiment, categorizes sentiment as 'positive,' 'negative,' or 'neutral' based on polarity scores.
Sentiment Analysis and DataFrame Modification: Sentiment scores are added to the DataFrame, and sentiment classification is applied. The result is a modified DataFrame with sentiment scores and classifications.


Text Preprocessing: Text data is prepared by removing HTML tags, expanding contractions, and normalizing text. Tokenization is performed using NLTK.
Bag of Words (BoW): A BoW representation is created using scikit-learn, and logistic regression is trained for sentiment prediction.
Averaged Word Embeddings: Averaged word embeddings are generated using spaCy's 'en_core_web_md' model, and logistic regression is used for sentiment prediction.
Mapping Ratings to Sentiments: A function maps numeric ratings to sentiment categories.


TextBlob Sentiment Analysis: Pearson correlation is calculated between user ratings and TextBlob sentiment scores.
Mapping User Ratings to Sentiment Categories (BoW Sentiment): User ratings are transformed into categorical sentiment labels based on thresholds.
Correlation with Categorized User Ratings (BoW Sentiment): Pearson correlation is computed between categorized user ratings and BoW sentiment predictions.
Averaged Word Embeddings Sentiment Analysis: Pearson correlation is calculated for averaged word embeddings and user sentiments.


VADER Sentiment Analysis: VADER sentiment scores are calculated for each review text.
Sentiment Distribution Comparison: Sentiment distribution is compared between TextBlob and VADER.
Word Cloud Generation: Word clouds are generated for different sentiment categories.


Attribute Extraction: Adjective and adverb phrases linked to attributes (cost/price, food, location, service) are extracted using NLTK.
Percentage of Correctly Worded Reviews: The percentage of correctly worded reviews is calculated, and Pearson correlations for negative and positive cases are explored.


Machine Learning Model for User Rating Prediction: Data cleaning, feature engineering, and model training are performed using Linear Support Vector Classification (Linear SVC), Random Forest, Multinomial Naive Bayes, Logistic Regression, and Decision Tree. Hyperparameter tuning is carried out to refine model performance.
