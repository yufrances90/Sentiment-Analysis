from torch import nn

class RNN(nn.Module):
    
    def __init__(
        self, 
        word_to_int, label_to_int, reviews, labels,
        hidden_dim=10, num_layers=3, dropout=0.5
    ):
        
        super(RNN, self).__init__()
        
        assert (word_to_int is not None and len(word_to_int) >= 10), "Invalid review vocabulary"
        assert (label_to_int is not None and len(label_to_int) >= 2), "Invalid label vocabulary"
        assert (reviews is not None and len(reviews) >= 100), "Invalid reviews"
        assert (labels is not None and len(labels) >= 100), "Invalid labels"
        assert (len(reviews) == len(labels)), "The length of reviews must be the same as the one of labels"
        
        self.preprocess_data(word_to_int, label_to_int, reviews)
        
        self.initialize_network(hidden_dim, num_layers, dropout)
        
    
    def preprocess_data(self, word_to_int, label_to_int, reviews):
        
        self.word_to_int = word_to_int
        self.label_to_int = label_to_int
        
        self.review_vocab = set(word_to_int.keys())
        self.label_vocab = set(label_to_int.keys())
        
        self.label_vocab_size = len(self.label_vocab)
        self.review_vocab_size = len(self.review_vocab)
        
        self.review_size = len(reviews)
        
    def initialize_network(self, hidden_dim, num_layers, dropout):
        
        self.input_dim = self.review_size
        self.embed_dim = self.review_vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = self.label_vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.word_embeddings = nn.Embedding(
            num_embeddings=self.input_dim, 
            embedding_dim=self.embed_dim
        )
        
        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.hidden_dim
        )
        
        self.fc2 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.output_dim
        )