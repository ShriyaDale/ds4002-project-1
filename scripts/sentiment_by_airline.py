import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data/airlines_reviews_with_sentiment.csv')

# Group by Airline and sentiment, then count reviews
sentiment_counts = df.groupby(['Airline', 'sentiment']).size().unstack(fill_value=0)

# Create bar graph
sentiment_counts.plot(kind='bar', figsize=(12, 6), width=0.8)
plt.title('Distribution of Reviews by Sentiment per Airline', fontsize=14, fontweight='bold')
plt.xlabel('Airline', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the graph
plt.savefig('airline_sentiment_bar_graph.png', dpi=300, bbox_inches='tight')
plt.close()

print("Bar graph saved as 'airline_sentiment_bar_graph.png'")
print("\nSummary of reviews by sentiment:")
print(sentiment_counts)
