#
# NLP Helper Functions
#
# Artificial Intelligence in Finance
# (c) Dr Yves J Hilpisch
# The Python Quants GmbH
#
import re
import nltk
import string
import pandas as pd
from pylab import plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from lxml.html.clean import Cleaner
from sklearn.feature_extraction.text import TfidfVectorizer
plt.style.use('seaborn-v0_8')

cleaner = Cleaner(style=True, links=True, allow_tags=[''],
                  remove_unknown_tags=False)

stop_words = stopwords.words('english')
stop_words.extend(['new', 'old', 'pro', 'open', 'menu', 'close'])


def remove_non_ascii(s):
    ''' Removes all non-ascii characters.
    '''
    return ''.join(i for i in s if ord(i) < 128)

def clean_up_html(t):
    t = cleaner.clean_html(t)
    t = re.sub('[\n\t\r]', ' ', t)
    t = re.sub(' +', ' ', t)
    t = re.sub('<.*?>', '', t)
    t = remove_non_ascii(t)
    return t

def clean_up_text(t, numbers=False, punctuation=False):
    ''' Cleans up a text, e.g. HTML document,
        from HTML tags and also cleans up the 
        text body.
    '''
    try:
        t = clean_up_html(t)
    except:
        pass
    t = t.lower()
    t = re.sub(r"what's", "what is ", t)
    t = t.replace('(ap)', '')
    t = re.sub(r"\'ve", " have ", t)
    t = re.sub(r"can't", "cannot ", t)
    t = re.sub(r"n't", " not ", t)
    t = re.sub(r"i'm", "i am ", t)
    t = re.sub(r"\'s", "", t)
    t = re.sub(r"\'re", " are ", t)
    t = re.sub(r"\'d", " would ", t)
    t = re.sub(r"\'ll", " will ", t)
    t = re.sub(r'\s+', ' ', t)
    t = re.sub(r"\\", "", t)
    t = re.sub(r"\'", "", t)    
    t = re.sub(r"\"", "", t)
    if numbers:
        t = re.sub('[^a-zA-Z ?!]+', '', t)
    if punctuation:
        t = re.sub(r'\W+', ' ', t)
    t = remove_non_ascii(t)
    t = t.strip()
    return t

def nltk_lemma(word):
    ''' If one exists, returns the lemma of a word.
        I.e. the base or dictionary version of it.
    '''
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
def tokenize(text, min_char=3, lemma=True, stop=True,
             numbers=False):
    ''' Tokenizes a text and implements some
        transformations.
    '''
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if len(t) >= min_char]
    if numbers:
        tokens = [t for t in tokens if t[0].lower()
                  in string.ascii_lowercase]
    if stop:
        tokens = [t for t in tokens if t not in stop_words]
    if lemma:
        tokens = [nltk_lemma(t) for t in tokens]
    return tokens

def generate_word_cloud(text, no, name=None, show=True):
    ''' Generates a word cloud bitmap given a
        text document (string).
        It uses the Term Frequency (TF) and
        Inverse Document Frequency (IDF) 
        vectorization approach to derive the
        importance of a word -- represented
        by the size of the word in the word cloud.
        
    Parameters
    ==========
    text: str
        text as the basis
    no: int
        number of words to be included
    name: str
        path to save the image
    show: bool
        whether to show the generated image or not
    '''
    tokens = tokenize(text)
    vec = TfidfVectorizer(min_df=2,
                      analyzer='word',
                      ngram_range=(1, 2),
                      stop_words='english'
                     )
    vec.fit_transform(tokens)
    wc = pd.DataFrame({'words': vec.get_feature_names(),
                       'tfidf': vec.idf_})
    words = ' '.join(wc.sort_values('tfidf', ascending=True)['words'].head(no))
    wordcloud = WordCloud(max_font_size=110,
                      background_color='white',
                      width=1024, height=768,
                      margin=10, max_words=150).generate(words)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    if name is not None:
        wordcloud.to_file(name)

def generate_key_words(text, no):
    try:
        tokens = tokenize(text)
        vec = TfidfVectorizer(min_df=2,
                      analyzer='word',
                      ngram_range=(1, 2),
                      stop_words='english'
                     )
    
        vec.fit_transform(tokens)
        wc = pd.DataFrame({'words': vec.get_feature_names(),
                       'tfidf': vec.idf_})
        words = wc.sort_values('tfidf', ascending=False)['words'].values
        words = [ a for a in words if not a.isnumeric()][:no]
    except:
        words = list()
    return words



