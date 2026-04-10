import pandas as pd
import random
from datetime import datetime, timedelta

def load_data(num_samples=1000):
    """
    Generates a synthetic dataset of social media fashion posts over the last 3 years.
    Returns a Pandas DataFrame.
    """
    trends = {
        "Y2K": ["love this y2k style!", "y2k is so outdated", "looking for y2k accessories", "is y2k coming back? yes!", "hate the low rise jeans y2k trend", "y2k aesthetic is my favorite"],
        "Streetwear": ["streetwear drop was fire", "streetwear is too expensive now", "classic streetwear look", "snagged some cool streetwear", "streetwear is dead", "best streetwear brand of the year"],
        "Vintage": ["found this amazing vintage jacket", "vintage shopping is hard but rewarding", "vintage clothes smell weird", "love the vintage vibe", "vintage fashion is sustainable", "not a fan of vintage"],
        "Minimalist": ["minimalist outfits are sleek", "minimalist fashion is boring", "clean minimalist look today", "building a minimalist wardrobe", "minimalist is too plain", "minimalist style saves time"],
        "Goth": ["pastel goth is so cute", "traditional goth vibes today", "nobody understands goth fashion", "goth style is too much for me", "love the dark goth aesthetic", "goth boots are uncomfortable"]
    }
    
    dates = []
    texts = []
    trend_labels = []
    locations_list = []
    
    locations = ["New York", "London", "Tokyo", "Paris", "Milan", "Los Angeles"]
    
    start_date = datetime.now() - timedelta(days=3 * 365) # 3 years ago
    
    for _ in range(num_samples):
        # Pick a random trend
        trend = random.choice(list(trends.keys()))
        
        # Pick a random sentence for that trend
        text = random.choice(trends[trend])
        
        # Add random noise/variations
        if random.random() > 0.7:
            text = text + " " + random.choice(["delivery was late", "shipping problems", "terrible quality", "cheap material", "wrong size", "small fit", "too large", "expensive price", "high cost", "fabric is bad", "great quality", "perfect size"])
            
        if random.random() > 0.7:
            text = text + " " + random.choice(["summer dress", "light fabric for hot weather", "going to a party", "night club outfit", "casual daily wear", "basic tees"])
            
        if random.random() > 0.5:
            text = text + f" #{trend.lower()}"
        
        # Generate a random date
        random_days = random.randint(0, 3 * 365)
        date = start_date + timedelta(days=random_days)
        
        # Generate a random location
        loc = random.choice(locations)
        
        dates.append(date)
        texts.append(text)
        trend_labels.append(trend)
        locations_list.append(loc)
        
    df = pd.DataFrame({
        "Date": dates,
        "Text": texts,
        "Trend": trend_labels,
        "Location": locations_list
    })
    
    return df

if __name__ == "__main__":
    df = load_data(10)
    print("Sample Data Generated:")
    print(df)
