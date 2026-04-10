import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download vader lexicon if not already downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """
        Analyzes the sentiment of a given text.
        Returns the sentiment label (Positive, Negative, Neutral) and the compound score.
        """
        scores = self.sia.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
            
        return label, compound

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    test_sentences = [
        "I absolutely love this new Y2K jacket!",
        "This streetwear drop is way too expensive and terrible quality.",
        "I bought a vintage shirt today."
    ]
    
    for sentence in test_sentences:
        label, score = analyzer.analyze_sentiment(sentence)
        print(f"Sentence: '{sentence}'")
        print(f"Sentiment: {label} (Score: {score:.4f})\n")
