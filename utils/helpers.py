import pandas as pd
import numpy as np
import re
import nltk
import string
import csv
import torch

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import \
    sent_tokenize, \
    word_tokenize
from tqdm.notebook import tqdm
from nltk.stem import WordNetLemmatizer 
from keras.preprocessing.text import Tokenizer
from torch.utils.data import TensorDataset, DataLoader

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
                
    @staticmethod
    def read_csv_into_pd_dataframe(key):
        return pd.read_csv(f'output/{key}.csv')
    
    @staticmethod
    def reviews_text_to_tensor(t, reviews):
        
        encoded_reviews = t.texts_to_matrix(reviews, mode='binary')
        
        return Utility.numpy_array_to_torch_tensor(encoded_reviews)
    
    @staticmethod
    def label_pd_series_to_tensor(labels):
        
        label_arr = Utility.pandas_series_to_numpy_array(labels).reshape(-1, 1)
        
        return Utility.numpy_array_to_torch_tensor(label_arr)
    
    @staticmethod
    def pandas_series_to_numpy_array(pd_series):
        return pd_series.to_numpy()
    
    @staticmethod
    def numpy_array_to_torch_tensor(np_arr):
        return torch.from_numpy(np_arr)
        
    @staticmethod
    def generate_tensors_from_csv_file(key, t=None):
        
        df = Utility.read_csv_into_pd_dataframe(key)
        
        reviews = df['reviews']
        labels = df['labels']
        
        if t is None and key == 'tn_reviews':
            
            t = Tokenizer()
            
            t.fit_on_texts(reviews)
            
        review_tensor = Utility.reviews_text_to_tensor(t, reviews)
        label_tensor = Utility.label_pd_series_to_tensor(labels)
        
        return {
            'tokenizer': t,
            'review_tensor': review_tensor,
            'label_tensor': label_tensor
        }
    
    @staticmethod
    def tensors_to_dataloader(r_tensor, l_tensor, batch_size=16):
        
        data = TensorDataset(r_tensor, l_tensor)
        data_loader = torch.utils.data.DataLoader(
            data, 
            shuffle=True,
            batch_size=batch_size
        )
        
        return data_loader
    
    @staticmethod
    def generate_dataloader_from_tensors(result):
        
        review_tensor = result['review_tensor']
        label_tensor = result['label_tensor']

        dataloader = Utility.tensors_to_dataloader(review_tensor, label_tensor)
        
        return dataloader