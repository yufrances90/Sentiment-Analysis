import re
import numpy as np
import torch

from collections import Counter
from tqdm import tqdm

class Utility:
    
    @staticmethod
    def escape_special_characters(reviews):
    
        updated_reviews = []

        for r in reviews:
            review = r.lower()
            review = re.sub('[^A-Za-z]+', ' ', review)
            updated_reviews.append(review)

        return updated_reviews

    @staticmethod
    def generate_review_vocab(reviews):

        total_counts = Counter()

        for review in reviews:
            for word in review.split():
                if len(word) > 2:
                    total_counts[word] = 1
    
        return Utility.generate_vocab_from_keys(total_counts.keys())

    @staticmethod
    def generate_label_vocab(labels):
        return Utility.generate_vocab_from_keys(labels)
    
    @staticmethod
    def generate_vocab_from_keys(keys):
        return list(sorted(set(keys)))

    @staticmethod
    def generate_text_to_int_dict(vocab):

        text_to_int = dict()

        for index, text in enumerate(vocab):
            text_to_int[text] = index

        return text_to_int
    
    @staticmethod
    def generate_review_tensor(reviews, review_vocab):
        
        word_to_int = Utility.generate_text_to_int_dict(review_vocab)
    
        review_arr = np.zeros((len(reviews), len(review_vocab)))

        for index, review in enumerate(tqdm(reviews)):
            for word in review.split():
                if word in review_vocab:
                    review_arr[index][word_to_int[word]] = 1

        return Utility.numpy_array_to_torch_tensor(review_arr)
    
    @staticmethod
    def numpy_array_to_torch_tensor(np_arr):
        return torch.from_numpy(np_arr)
    
    @staticmethod
    def pandas_series_to_torch_tensor(pd_series):
        return Utility.numpy_array_to_torch_tensor(pd_series.to_numpy())
    
    @staticmethod
    def generate_review_int_dict(reviews):
        
        review_vocab = Utility.generate_review_vocab(reviews)
        
        return Utility.generate_text_to_int_dict(review_vocab)
    
    
    @staticmethod
    def generate_label_int_dict(labels):
        
        label_vocab = Utility.generate_label_vocab(labels)
        
        return Utility.generate_text_to_int_dict(label_vocab)