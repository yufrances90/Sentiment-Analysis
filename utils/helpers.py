import pandas as pd
import numpy as np
import re
import nltk
import string
import csv

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import \
    sent_tokenize, \
    word_tokenize
from tqdm.notebook import tqdm
from nltk.stem import WordNetLemmatizer 

translator = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words("english")) 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

class Utility:
    
    @staticmethod
    def preprocess_review(reviews):
    
        updated_reviews = []

        for review  in tqdm(reviews):

            review = review.lower()

            review = re.sub(r'\d+', '', review)

            review = review.translate(translator)

            review = " ".join(review.split())

            word_tokens = word_tokenize(review) 

            filtered_words = [word for word in word_tokens if word not in stop_words] 
            lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in filtered_words] 

            review = " ".join(lemmas) 

            updated_reviews.append(review)
            
        return updated_reviews
    
    @staticmethod
    def write_list_to_csv_file(reviews, labels, filepath):
        
        with open(f'output/{filepath}', 'w', newline='') as csvfile:
            
            fieldnames = ['reviews', 'labels']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for (review, label) in tqdm(zip(reviews, labels)):
                writer.writerow({
                    fieldnames[0]: review, 
                    fieldnames[1]: label
                })

   