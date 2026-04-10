import os
import sys
import io

# Force UTF-8 encoding for stdout to support emojis on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from data_loader import load_data
from sentiment_analyzer import SentimentAnalyzer
from trend_detector import TrendDetector
from advanced_analytics import AdvancedAnalytics

def main():
    print("--- Fashion Trend Prediction & Sentiment Analysis ---")
    
    # 1. Load/Generate Data
    print("\n[1/4] Generating synthetic social media data (3 years history)...")
    num_samples = 2000
    df = load_data(num_samples=num_samples)
    print(f"Generated {len(df)} posts.")

    # 2. Analyze Sentiment
    print("\n[2/4] Analyzing sentiments for all posts...")
    analyzer = SentimentAnalyzer()
    
    sentiment_labels = []
    sentiment_scores = []
    for text in df['Text']:
        label, score = analyzer.analyze_sentiment(text)
        sentiment_labels.append(label)
        sentiment_scores.append(score)
        
    df['Sentiment_Label'] = sentiment_labels
    df['Sentiment_Score'] = sentiment_scores
    
    # Display sample output matching positive, negative, neutral
    print("\nSample Sentiment Detection:")
    sample_outputs = df.sample(5)[['Text', 'Trend', 'Sentiment_Label', 'Sentiment_Score']]
    for _, row in sample_outputs.iterrows():
        print(f"Text: '{row['Text']}' | Label: {row['Sentiment_Label']} | Score: {row['Sentiment_Score']:.2f}")

    # 3. Trend Detection (Weekly, Monthly and Yearly)
    print("\n[3/4] Performing weekly, monthly and yearly trend detections...")
    detector = TrendDetector(df)
    
    weekly_trends = detector.weekly_trend_detection()
    monthly_trends = detector.monthly_trend_detection()
    yearly_trends = detector.yearly_trend_detection()

    print("\nSample Weekly Aggregation:")
    print(weekly_trends.head())
    
    print("\nSample Monthly Aggregation:")
    print(monthly_trends.head())
    
    print("\nSample Yearly Aggregation:")
    print(yearly_trends.head())

    # 4. Advanced Analytics (Issue, Theme, Repeat Complaint Detection)
    print("\n[4/5] Running Advanced Analytics...")
    analytics = AdvancedAnalytics(df)
    analytics.detect_issues()
    analytics.detect_themes()
    analytics.detect_repeat_complaints()
    analytics.detect_locations()

    # 5. Visualization
    output_dir = "results"
    print(f"\n[5/5] Generating and saving trend plots to '{output_dir}/'...")
    detector.plot_trends(output_dir=output_dir)
    
    print("\n--- Project Execution Completed! ---")
    print(f"Check the '{output_dir}' directory for the generated PNG plots showing trend prediction.")

if __name__ == "__main__":
    main()
