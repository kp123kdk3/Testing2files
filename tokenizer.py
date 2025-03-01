from collections import Counter
import re

class SimpleTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        
    def encode(self, text):
        # Simple character-level tokenization for demo
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, tokens):
        # Convert tokens back to text
        return ''.join([chr(t) for t in tokens])

    def train(self, texts):
        # Initialize with special tokens
        self.token_to_id = self.special_tokens.copy()
        
        # Tokenize and count words
        words = []
        for text in texts:
            words.extend(re.findall(r'\b\w+\b', text.lower()))
        
        # Get most common words
        word_counts = Counter(words)
        common_words = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        # Create vocabulary
        for i, (word, _) in enumerate(common_words):
            self.token_to_id[word] = i + len(self.special_tokens)
            
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
    def encode(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        return [self.token_to_id.get(word, self.special_tokens['<UNK>']) for word in words]
        
    def decode(self, ids):
        return ' '.join([self.id_to_token.get(id, '<UNK>') for id in ids]) 