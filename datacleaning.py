import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import datetime
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

nltk.download('vader_lexicon', quiet=True)

DATE_START = datetime.date(2014, 1, 1)
DATE_END = datetime.date(2018, 12, 31)
TICKER = 'aapl'
FORECAST_DAYS = 1
def date_range(start, end):
    return [start + datetime.timedelta(days=i) for i in range((end - start).days + 1)]

def analyze_sentiment(headlines):
    return [sentiment_analyzer.polarity_scores(sentence) for sentence in headlines]

sentiment_analyzer = SentimentIntensityAnalyzer()
custom_lexicon = {}
stock_lexicon = pd.read_csv('/include/stock_lex.csv')
stock_lexicon['Sentiment'] = (stock_lexicon['Aff_Score'] + stock_lexicon['Neg_Score']) / 2
stock_dict = {word: score for word, score in zip(stock_lexicon['Item'], stock_lexicon['Sentiment']) 
              if len(word.split()) == 1}
max_pos = max(v for v in stock_dict.values() if v > 0)
min_neg = min(v for v in stock_dict.values() if v < 0)

for word, score in stock_dict.items():
    if score > 0:
        custom_lexicon[word] = score / max_pos * 4
    else:
        custom_lexicon[word] = score / min_neg * -4

# Add financial domain lexicons
for word in pd.read_csv('/include/positive.csv', header=None)[0]:
    custom_lexicon[word.lower()] = 2

for word in pd.read_csv('/include/negative.csv', header=None)[0]:
    custom_lexicon[word.lower()] = -2

# Update the sentiment analyzer with enhanced lexicon
custom_lexicon.update(sentiment_analyzer.lexicon)
sentiment_analyzer.lexicon = custom_lexicon

def process_news():
    news_df = pd.read_csv(f'/files/{TICKER}_News_All.csv')
    news_df['Headline'] = news_df['Headline'].str.lower().str.replace(r'[^\w\s]+', '')
    sentiment_df = pd.DataFrame(analyze_sentiment(news_df['Headline']))
    news_df = pd.concat([news_df, sentiment_df], axis=1)
    news_df = news_df.drop(['neg', 'pos', 'neu'], axis=1)
    news_df = news_df[~((news_df['compound'] >= -0.05) & (news_df['compound'] <= 0.05))]
    news_df['Sentiment'] = news_df['compound'].apply(lambda x: "Positive" if x > 0 else "Negative")
    for source in ['FT', 'NYTimes', 'BS']:
        news_df[f'compound{source}'] = news_df['compound'][news_df['Source'] == source]
    
    news_df = news_df.drop(['compound'], axis=1)
    visualize_sentiment(news_df)
    
    return news_df

def visualize_sentiment(news_df):
    plt.figure(figsize=(20, 6))
    plt.hist(news_df['compound{}'.format(news_df['Source'])], bins=100)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentiment Scores')
    plt.show()
    news_df.groupby(['Source', 'Sentiment']).size().unstack().plot(kind='bar', figsize=(10, 5))
    plt.ylabel('Frequency')
    plt.title('Sentiment by News Source')
    plt.show()
    neg_count = news_df[news_df['Sentiment'] == 'Negative']['Headline'].count()
    pos_count = news_df[news_df['Sentiment'] == 'Positive']['Headline'].count()
    print(f"Negative Sentiment News: {neg_count}")
    print(f"Positive Sentiment News: {pos_count}")
    print(f"Total News: {neg_count + pos_count}")
    negative_text = ' '.join(news_df[news_df['Sentiment'] == 'Negative']['Headline'])
    wordcloud = WordCloud(width=600, height=400, background_color='black').generate(negative_text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Common Words in Negative Headlines')
    plt.show()

def prepare_features():
    news_df = process_news()
    daily_sentiment = news_df.groupby(['Date']).mean()
    daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
    daily_sentiment = daily_sentiment[['compoundFT', 'compoundNYTimes', 'compoundBS']]
    stock_df = pd.read_csv(f'files/{TICKER}_Stock_Data.csv')
    stock_df = stock_df.set_index('Date')
    stock_df.index = pd.to_datetime(stock_df.index)
    date_index = pd.DataFrame(index=pd.to_datetime(date_range(DATE_START, DATE_END)))
    
    # Merge all data
    merged_df = date_index.join([daily_sentiment, stock_df])
    
    # Handle missing values
    for col in ['compoundFT', 'compoundNYTimes', 'compoundBS']:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    merged_df[["Close", "Volume", "Adj Close"]] = merged_df[["Close", "Volume", "Adj Close"]].interpolate()
    merged_df['Target'] = merged_df['Adj Close'].shift(-FORECAST_DAYS)
    add_technical_indicators(merged_df)
    cols_to_keep = ['compoundFT', 'compoundNYTimes', 'compoundBS', 
                     'Close', 'Volume', 'Adj Close', 'RSI', 'MACD', 'Target']
    final_df = merged_df[cols_to_keep].dropna()
    
    # Save processed data
    final_df.to_csv(f'files/{TICKER}_features.csv')
    
    # Show feature importance
    show_feature_importance(final_df)
    
    return final_df

def add_technical_indicators(df):
    # MACD
    df['EMA12'] = df['Adj Close'].ewm(span=12).mean()
    df['EMA26'] = df['Adj Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    
    # RSI
    delta = df['Adj Close'].diff()
    gain = delta.mask(delta < 0, 0)
    loss = -delta.mask(delta > 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

def show_feature_importance(df):
    corr = df.corr()['Target'].sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    corr.drop('Target').plot(kind='bar')
    plt.title('Feature Correlation with Target')
    plt.ylabel('Correlation Coefficient')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Execute the feature engineering process
features = prepare_features()
print(f"Feature engineering complete. Dataset shape: {features.shape}")