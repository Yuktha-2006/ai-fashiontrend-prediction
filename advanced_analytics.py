import pandas as pd
from collections import Counter

class AdvancedAnalytics:
    def __init__(self, df):
        """
        Initializes with a Pandas DataFrame containing a 'Text' column.
        Creates a 'clean_text' column used by the analytics methods.
        """
        self.df = df
        # Create 'clean_text' for analysis
        if 'Text' in self.df.columns:
            self.df['clean_text'] = self.df['Text'].astype(str).str.lower()
        else:
            self.df['clean_text'] = ""

    def detect_issues(self):
        """
        Groups complaints into categories and counts occurrences.
        """
        print("\n--- Issue Detection System ---")
        issues = {
            "delivery": ["delivery", "shipping", "late"],
            "quality": ["quality", "material", "fabric"],
            "size": ["size", "fit", "small", "large"],
            "price": ["price", "expensive", "cost"]
        }

        issue_counts = {}
        for issue, words in issues.items():
            count = self.df['clean_text'].apply(
                lambda x: any(word in x for word in words)
            ).sum()
            issue_counts[issue] = count
            print(f"{issue} : {count}")

        if issue_counts and sum(issue_counts.values()) > 0:
            biggest_issue = max(issue_counts, key=issue_counts.get)
            print(f"👉 Now you can say: ✔ “{biggest_issue.capitalize()} is the biggest issue” 🔥")
            
        return issue_counts

    def detect_themes(self):
        """
        Groups mentions into fashion themes and counts occurrences.
        """
        print("\n--- Fashion Theme Detection ---")
        themes = {
            "summer": ["summer", "light", "hot"],
            "party": ["party", "night", "club", "glam"],
            "casual": ["casual", "daily", "basic"]
        }

        theme_counts = {}
        for theme, words in themes.items():
            count = self.df['clean_text'].str.contains('|'.join(words), na=False).sum()
            theme_counts[theme] = count
            print(f"{theme} : {count}")

        if theme_counts and sum(theme_counts.values()) > 0:
            biggest_theme = max(theme_counts, key=theme_counts.get)
            print(f"👉 🎯 Now you can say: ✔ “{biggest_theme.capitalize()} wear is most discussed category”")
            
        return theme_counts

    def detect_repeat_complaints(self):
        """
        Detects if the same issue appears frequently based on the most common words.
        """
        print("\n--- Repeat Complaint Detection ---")
        all_words = " ".join(self.df['clean_text']).split()

        # Custom stop words filter to get more meaningful "issues"
        stop_words = {"this", "is", "so", "for", "my", "the", "a", "too", "but", "not", "of", "to", "and", "in", "it", "was"}
        filtered_words = [word for word in all_words if word not in stop_words and len(word) > 2 and not word.startswith("#")]

        common_words = Counter(filtered_words).most_common(10)
        
        print("Top 10 most common words:")
        print(common_words)
        
        if common_words:
            top_word = common_words[0][0]
            print(f"👉 Insight: ✔ “{top_word} appears frequently → recurring issue”")
            
        return common_words

    def detect_locations(self):
        """
        Analyzes the newly added Location data to find geographic trend insights.
        """
        if 'Location' not in self.df.columns:
            return
            
        print("\n--- Geographic Location Tracking ---")
        
        if 'Trend' in self.df.columns:
            trend_counts = self.df['Trend'].value_counts()
            top_trend = trend_counts.index[0]
            
            top_trend_data = self.df[self.df['Trend'] == top_trend]
            location_counts = top_trend_data['Location'].value_counts()
            
            top_location = location_counts.index[0]
            bottom_location = location_counts.index[-1]
            
            print(f"👉 Insight: ✔ “The {top_trend} trend is spiking in {top_location}, but declining in {bottom_location}.”")
            
            print("\nMentions Breakdown:")
            # Simple aggregation to show
            cross_tab = pd.crosstab(self.df['Trend'], self.df['Location'])
            print(cross_tab)

if __name__ == "__main__":
    test_df = pd.DataFrame({
        "Text": [
            "the delivery was late", 
            "bad quality fabric", 
            "summer dress is hot", 
            "shipping problem with standard delivery",
            "casual daily wear"
        ]
    })
    analytics = AdvancedAnalytics(test_df)
    analytics.detect_issues()
    analytics.detect_themes()
    analytics.detect_repeat_complaints()
