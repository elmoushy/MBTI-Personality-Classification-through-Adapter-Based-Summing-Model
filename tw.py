import csv
import nltk
from nltk.corpus import stopwords

def make_text_upper(text):
    return text.upper()

from nltk.stem import SnowballStemmer

def stem_text1(text):
    stemmer = SnowballStemmer('english')
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text

def remove_stopwords(text):
    """
    Removes stop words from a given text.
    """
    tokens = nltk.word_tokenize(text.lower())
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    filtered_text = " ".join(filtered_tokens)
    
    return filtered_text

def to_lowercase(text):
    """
    Converts a string to lowercase.
    """
    return text.lower()

def clean_text(text):
    text=to_lowercase(text)
    import re
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_special_chars(text):
    import re
    special_chars = r'[\~\`\!\@\#\$\%\^\&\*\(\)\-\_\+\=\{\}\[\]\|\:\;\"\'\<\>\?\,\.\/]'
    return re.sub(special_chars, '', text)

import re

def remove_non_english_letters(text):
    """
    Removes any non-English letters from a string using regular expressions while preserving spaces.
    
    Args:
    text (str): The input string to be processed.
    
    Returns:
    str: The processed string with only English letters and spaces.
    """
    pattern = re.compile('[^a-zA-Z\s]')
    english_text = re.sub(pattern, '', text)
    return english_text

import langdetect

import nltk
from nltk.stem import PorterStemmer

def stem_text(text):
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(text)
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    return stemmed_text


def remove_repeated_letters_words(text):
    try:
        lang = langdetect.detect(text)
    except langdetect.lang_detect_exception.LangDetectException:
        lang = "unknown"
    if lang == "en":
        pattern = r"\b([a-zA-Z])\1+\b"
        return re.sub(pattern, "", text)
    else:
        return text
    
def replace_repeated_letters(text):

    words = text.split()
    new_words = []
    for word in words:
        new_word = ""
        last_char = None
        count = 0
        for char in word:
            if char == last_char:
                count += 1
                if count <= 2:
                    new_word += char
            else:
                new_word += char
                last_char = char
                count = 1
        new_words.append(new_word)
    return " ".join(new_words)

import nltk
from nltk.stem import PorterStemmer

def create_poster(text):
    words = nltk.word_tokenize(text)
    
    stemmer = PorterStemmer()
    
    stemmed_words = [stemmer.stem(word) for word in words]
    
    stemmed_text = ' '.join(stemmed_words)
        
    return stemmed_text

def expand_contractions(text):
    contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I had",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "shan't": "shall not",
    "she'd": "she had",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they had",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you had",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}
    
    for contraction, expanded in contractions.items():
        text = text.replace(contraction, expanded)
    
    return text

import spacy

def lemmatize_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    lemmas_text = ' '.join(lemmas)
    return lemmas_text

english_words = set(nltk.corpus.words.words()) 
def remove_non_english_wordsssssssss(text):
    words = text.split()  
    english_words_in_text = [word for word in words if word.lower() in english_words]  
    return ' '.join(english_words_in_text)  

import enchant
def remove_nonenglish_words(text):
    english_dict = enchant.Dict("en_US")
    words = text.split()
    filtered_words = [word for word in words if english_dict.check(word)]
    return ' '.join(filtered_words)

def expand_s(text):
    
    text = text.replace("'s", " is")
    
    return 

def count_words(text):
    words = text.split()
    return len(words)
i=0
tot=0

with open('Dataset_summarized11_1label-16.csv', mode='r') as input_file, open('D1111.csv', mode='a', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    for row in reader:
        cleaned_text = count_words(row[1])
        tot+=cleaned_text
        i+=1
        cleaned_text = remove_special_chars(cleaned_text)
        #cleaned_text = remove_non_english_letters(cleaned_text)
        row[1] = cleaned_text
        writer.writerow(row)
avg=tot/i
print(avg)