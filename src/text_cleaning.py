import re
import string

class TextCleaner:
    def __init__(self):
        self.contraction_mapping = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
            "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'll": "how will",
            "how's": "how is", "i'd": "i would", "i'll": "i will", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would", "it'll": "it will",
            "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not",
            "mightn't": "might not", "mustn't": "must not", "needn't": "need not",
            "oughtn't": "ought not", "shan't": "shall not", "sha'n't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is", "shouldn't": "should not",
            "that'd": "that would", "that's": "that is", "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are", "they've": "they have", "wasn't": "was not",
            "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "are not", "what'll": "what will", "what're": "what are", "what's": "what is",
            "what've": "what have", "where's": "where is", "who'll": "who will", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
            "you're": "you are", "you've": "you have"
        }

        self.stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "he", "him", "his", "she", "her", "it", "its", "they", "them", 
            "their", "what", "which", "who", "whom", "this", "that", "am", "is", "are", 
            "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
            "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", 
            "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", 
            "between", "into", "through", "during", "before", "after", "above", "below", 
            "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
            "further", "then", "once", "here", "there", "all", "any", "both", "each", "few", 
            "more", "most", "other", "some", "such", "own", "same", "so", "than", "too", "very",
            "st", "nd", "rd", "th", "amazon", "prime",
            "don", "ve", "couldn", "cant", "dont", "wont", "shouldn", "didnt", "doesnt"
        }

    def expand_contractions(self, text):
        contraction_pattern = re.compile(r'\b(' + '|'.join(self.contraction_mapping.keys()) + r')\b')
        def replace_match(match):
            return self.contraction_mapping[match.group(0)]
        return contraction_pattern.sub(replace_match, text)

    def clean_review_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        
        if text == "review text not found":
            return ""
            
        text = self.expand_contractions(text)
        
        text = re.sub(r'([.,!?()""])', r' \1 ', text)
        
        text = re.sub(r'https?://\S+|www\.\S+|\S+\.[a-z]{2,3}', '', text)
        
        text = re.sub(r'\b\d+(st|nd|rd|th)\b', '', text)
        
        text = re.sub(r'\d+', '', text)
        
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        punctuation_pattern = r'[{}]'.format(re.escape(string.punctuation))
        text = re.sub(punctuation_pattern, ' ', text)
        
        words = text.split()
        cleaned_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return " ".join(cleaned_words)