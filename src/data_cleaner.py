import re
import string

ENGLISH_STOP_WORDS = frozenset("""
a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can't cannot could couldn't
did didn't do does doesn't doing don't down during
each
few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
let's
me more most mustn't my myself
no nor not
of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too
under until up
very
was wasn't we we'd we'll we're we_ve were weren't what what's when
when's where where's which while who who's whom why why's with won't
would wouldn't
you you'd you'll you're you_ve your yours yourself yourselves
""".split())

class TextCleaner:
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\S+\.[a-z]{2,3}")
    ORDINAL_PATTERN = re.compile(r"\b\d+(st|nd|rd|th)\b")
    DIGIT_PATTERN = re.compile(r"\d+")
    PUNCTUATION_SPACE_PATTERN = re.compile(r'([.,!?()""])')
    MULTI_SPACE_PATTERN = re.compile(r"\s+")

    def __init__(self):
        self.contractions = {
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
        negations = {"no", "not", "nor", "neither", "never", "dont", "cant", "wont", "didnt", "doesnt", "cannot"}
        base_stops = ENGLISH_STOP_WORDS.union({"amazon", "prime", "st", "nd", "rd", "th"})
        self.stop_words = base_stops.difference(negations)
        self.contraction_regex = re.compile(r'\b(' + '|'.join(self.contractions.keys()) + r')\b')

    def expand(self, text: str) -> str:
        return self.contraction_regex.sub(lambda m: self.contractions[m.group(0)], text)

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.lower().strip()
        if text == "review text not found" or not text:
            return ""
            
        text = text.replace("’", "'")
        text = self.expand(text)
        text = self.PUNCTUATION_SPACE_PATTERN.sub(r' \1 ', text)
        text = self.URL_PATTERN.sub('', text)
        text = self.ORDINAL_PATTERN.sub('', text)
        text = self.DIGIT_PATTERN.sub('', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        punctuation_pattern = r'[{}]'.format(re.escape(string.punctuation))
        text = re.sub(punctuation_pattern, ' ', text)
 
        tokens = text.split()
        transformed_tokens = []
        negation_countdown = 0
        negation_anchors = {"no", "not", "nor", "neither", "never", "cannot", "lacks", "failed"}
        
        for word in tokens:
            if word in negation_anchors:
                negation_countdown = 3
                if word not in self.stop_words and len(word) > 1:
                    transformed_tokens.append(word)
                continue
            
            if negation_countdown > 0:
                word = f"not_{word}"
                negation_countdown -= 1
                
            if word not in self.stop_words and len(word) > 1:
                transformed_tokens.append(word)
        
        return self.MULTI_SPACE_PATTERN.sub(' ', " ".join(transformed_tokens)).strip()