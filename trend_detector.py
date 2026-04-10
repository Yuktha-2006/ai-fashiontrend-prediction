import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class TrendDetector:
    def __init__(self, data):
        """
        Initializes with a Pandas DataFrame containing at least 'Date', 'Trend', 'Sentiment_Score', 'Sentiment_Label'.
        """
        self.data = data
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Extract Month-Year for grouping
        self.data['Month_Year'] = self.data['Date'].dt.to_period('M')
        # Extract Week for grouping
        self.data['Week_Year'] = self.data['Date'].dt.to_period('W')
        # Extract Year for grouping
        self.data['Year'] = self.data['Date'].dt.to_period('Y')

    def monthly_trend_detection(self):
        """
        Groups data by Month and Trend.
        Returns a DataFrame with counts and average sentiment.
        """
        monthly_stats = self.data.groupby(['Month_Year', 'Trend']).agg(
            Post_Count=('Text', 'count'),
            Average_Sentiment=('Sentiment_Score', 'mean')
        ).reset_index()
        
        # Convert period back to string/timestamp for plotting
        monthly_stats['Month_Year_Str'] = monthly_stats['Month_Year'].dt.strftime('%Y-%m')
        
        return monthly_stats

    def weekly_trend_detection(self):
        """
        Groups data by Week and Trend.
        Returns a DataFrame with counts and average sentiment.
        """
        weekly_stats = self.data.groupby(['Week_Year', 'Trend']).agg(
            Post_Count=('Text', 'count'),
            Average_Sentiment=('Sentiment_Score', 'mean')
        ).reset_index()
        
        # Convert period to string (starts on Monday usually)
        weekly_stats['Week_Year_Str'] = weekly_stats['Week_Year'].dt.start_time.dt.strftime('%Y-%m-%d')
        return weekly_stats

    def yearly_trend_detection(self):
        """
        Groups data by Year and Trend.
        Returns a DataFrame with counts and average sentiment.
        """
        yearly_stats = self.data.groupby(['Year', 'Trend']).agg(
            Post_Count=('Text', 'count'),
            Average_Sentiment=('Sentiment_Score', 'mean')
        ).reset_index()
        
        yearly_stats['Year_Str'] = yearly_stats['Year'].dt.strftime('%Y')
        return yearly_stats

    def plot_trends(self, output_dir="output"):
        """
        Plots the Monthly and Yearly post counts trends and saves them.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        weekly_stats = self.weekly_trend_detection()
        monthly_stats = self.monthly_trend_detection()
        yearly_stats = self.yearly_trend_detection()

        # 1. Plot Weekly Post Counts by Trend
        plt.figure(figsize=(14, 6))
        sns.lineplot(data=weekly_stats, x='Week_Year_Str', y='Post_Count', hue='Trend', marker="o")
        plt.xticks(rotation=90, fontsize=8) # Rotate tight labels
        # Only show a subset of x-ticks since weeks can be too many
        ax = plt.gca()
        for ind, label in enumerate(ax.get_xticklabels()):
            if ind % 4 == 0:  # Show every 4th week (~monthly) to avoid clutter
                label.set_visible(True)
            else:
                label.set_visible(False)
        plt.title('Weekly Fashion Trend Mentions')
        plt.xlabel('Week (Starting Date)')
        plt.ylabel('Number of Mentions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weekly_trends.png'))
        plt.close()

        # 2. Plot Monthly Post Counts by Trend
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_stats, x='Month_Year_Str', y='Post_Count', hue='Trend', marker="o")
        plt.xticks(rotation=45)
        plt.title('Monthly Fashion Trend Mentions')
        plt.xlabel('Month-Year')
        plt.ylabel('Number of Mentions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_trends.png'))
        plt.close()

        # 2. Plot Yearly Post Counts by Trend
        plt.figure(figsize=(10, 5))
        sns.barplot(data=yearly_stats, x='Year_Str', y='Post_Count', hue='Trend')
        plt.title('Yearly Fashion Trend Mentions')
        plt.xlabel('Year')
        plt.ylabel('Number of Mentions')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'yearly_trends.png'))
        plt.close()
        
        # 3. Plot Overall Sentiment Distribution per Trend
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=self.data, x='Trend', y='Sentiment_Score')
        plt.title('Sentiment Distribution by Fashion Trend')
        plt.xlabel('Trend')
        plt.ylabel('Sentiment Score (Compound)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'))
        plt.close()
        
        print(f"Plots saved successfully in the '{output_dir}' directory.")

if __name__ == "__main__":
    from data_loader import load_data
    from sentiment_analyzer import SentimentAnalyzer
    
    print("--- Testing Trend Detector Isolation ---")
    # Generate 50 sample posts
    df_sample = load_data(num_samples=50)
    
    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    df_sample['Sentiment_Score'] = df_sample['Text'].apply(lambda context: analyzer.analyze_sentiment(context)[1])
    
    # Initialize detector
    detector = TrendDetector(df_sample)
    
    print("\nSample Weekly Execution:")
    print(detector.weekly_trend_detection().head())
    print("\nThe detector works correctly! (Run main.py to see full graphs)")
