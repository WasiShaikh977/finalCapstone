# Importing the required libraries
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

# Loading the nlp model we want to use and the spacy text blob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

# Reading in the csv file
df = pd.read_csv('Amazon_reviews.csv')

def sentiment_analysis(df):
    '''
    This function reads through the reviews in our dataset and assigns them a sentiment score.
    -1 indicates extremely negative sentiment, 1 indicates extremely positive & indicates neutral sentiment.
    The input to this function is simpy the right dataset.
    '''
    # We make the clean_data variable global as we are going to call it later outside of this function as well
    global clean_data
    # Preprocessing steps
    # Dropping NA values
    clean_data = df.dropna(subset=['reviews.text'])
    # clean_data = clean_data[0:501] # Was used for trial and error when writing the code
    
    # Creating an empty list to store the sentiment values
    sentiment_list = []
    # Creating a for loop to loop through every reveiw in our data
    for text in clean_data['reviews.text']:
        # More preprocessing steps to remove unwanted spaces and standardize ecerything into small caps
        text_in =str(text.lower().strip())
        doc = nlp(text_in)
        # Using the .is_stop method to remove the stop words for our analysis
        filtered_text = ' '.join(token.text for token in doc if not token.is_stop)
        # We need to classify the sentiments as positive, negative or neutral before appending it to our list
        # Since the range lies between -1 and 1 we will label anything above 0.2 as positive, anything less than -0.2 as negative
        # and anything between -0.2 and 0.2 as Neutral
        if doc._.polarity > 0.2:
            sentiment_list.append("Positive")
        elif doc._.polarity < -0.2:
            sentiment_list.append("Negative")
        else:
            sentiment_list.append("Neutral")
    # Finally inserting our sentiment values into our dataframe
    clean_data['Sentiment'] = sentiment_list
    # print(clean_data[['reviews.text','Sentiment']])

# Running our function
sentiment_analysis(df)


# Creating another function to check out random samples and their sentiment values from our datase
def random_sampler(x,y,z):
    '''
    Enter the 3 random row values to get the review and the sentiment value from our data.
    Please do not exceed the max count from our dataset.
    This function won't work if the sentiment_analysis function hasn't been run at least once.
    '''
    # Creating a list of the random numbers to loop through
    row_num_list = [x,y,z]
    # Simple for loop that prints out the sample number, the review and the sentiment
    for num in row_num_list:
        print('This is sample number',num)
        print(clean_data.loc[num,'reviews.text'])
        print(clean_data.loc[num,'Sentiment'])
        print('\n')
random_sampler(23, 18, 100)
print('\n')
# As we can see in the previous example, the sentiment analysis function seems to have done pretty well for this task. 
# Sample no. 23 in our dataset is a negative review and our model has deemed it to be -0.278 which is exactly what we want.
# Sample 18 is neutral/slighlty positive so it got a rating of 0.1699 and
# sample 100 is a positive review and the model gave it a rating of 0.458.


# Next let's compare our negative review (#23) to our positive review (#100),
# the result should be that the reviews are dissimmilar

doc1 = nlp(clean_data.loc[23,'reviews.text'])
doc2 = nlp(clean_data.loc[100,'reviews.text'])


# Calculating similarity 
similarity_score = doc1.similarity(doc2)
print(similarity_score)
# We get a similarity score of 0.39 which is not the best possible answer but it is closer to 0 than it is to
# 1 suggesting the model undestands these 2 reviews are dissimilar
