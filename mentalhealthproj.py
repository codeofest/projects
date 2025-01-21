import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('punkt_tab')  # Ensure this is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize stopwords list and lemmatizer
stopwords = set(stopwords.words('english'))  # English stopwords
lemmatizer = WordNetLemmatizer()

def analysetext(txt):
    # Step 1: Preprocess the text
    pptext = preprocess(txt)
    # Step 2: Analyze sentiment
    sscore = analysesentiment(pptext)
    # Step 3: Extract keywords
    distresskeywd = extractkeywords(pptext, diskeywdlist)

    # Risk level classification
    if sscore > 0.5 and not distresskeywd:
        rlevel = "LOW"
    elif sscore <= 0.5 and len(distresskeywd) < 3:
        rlevel = "MODERATE"
    else:
        rlevel = "HIGH"

    return rlevel

def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenize text
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return " ".join(tokens)

def analysesentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]

def extractkeywords(text, diskeywdlist):
    return [word for word in text.split() if word in diskeywdlist]

# Distress-related keywords
diskeywdlist = [
    "Sad", "Hopeless", "Helpless", "Miserable", "Empty", "Lonely", "Isolated",
    "Tired", "Overwhelmed", "Exhausted", "Numb", "Worthless", "Guilty",
    "Unloved", "Broken", "Lost", "Ashamed", "Defeated", "Angry", "Frustrated"
]

# User input
txtinput = str(input("How are you feeling today? "))
result = analysetext(txtinput)
print(result)
