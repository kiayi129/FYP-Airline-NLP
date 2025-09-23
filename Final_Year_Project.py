# ============================
#      IMPORTS AND SETUP
# ============================
import pandas as pd
import csv
import re
import pandas as pd
import nltk
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim as gensimvis
from nltk.tokenize import word_tokenize
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
from gensim.models import CoherenceModel
from transformers import pipeline
from transformers import AutoTokenizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from collections import Counter

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ============================
# Function 1: Data Loading and Preprocessing
# ============================

# Load dataset
def load_reviews(file_path):
    """Load and preprocess reviews from CSV."""
    print("Loading British Airline dataset...")
    df = pd.read_csv(file_path)
    print("British Airways dataset loaded successfully! ")
    print("\nCleaning Dataset...") 
    columns_to_drop = ['header', 'author', 'place', 'aircraft', 'traveller_type', 'seat_type', 'route', 'date_flown',
                       'recommended', 'trip_verified', 'rating', 'seat_comfort', 'cabin_staff_service', 'food_beverages',
                       'ground_services', 'value_for_money', 'entertainment']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    df['cleaned_tokens'] = df['content'].apply(clean_text)
    print("Dataset has been cleaned successfully!")
    return df

# Text cleaning and tokenization
custom_stopwords = set(stopwords.words('english'))
custom_stopwords.update(["one", "get", "would", "could", "told", "good", "heathrow", "airline", "even", "terminal", "london", "take", "really"])
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and tokenize text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords and len(word) > 2]
    return tokens

# ============================
# Function 2: Bigram Detection and Frequency Analysis
# ============================

# Bigram detection (PMI Calculation)
def build_phrases(tokenized_reviews):
    # Create bigram model
    print("\nBuilding bigram model...")
    bigram = Phrases(tokenized_reviews, min_count=5, threshold=10)
    bigram_model = Phraser(bigram)
    
    # Apply bigram model to reviews 
    print("Applying bigram model...")
    tokenized_reviews = [bigram_model[review] for review in tokenized_reviews]
    
    # Print some results to verify bigram
    print("\nExample of bigram transformation:")
    for i in range(3):  # Show first 3 reviews after bigram processing
        print(f"Processed: {tokenized_reviews[i]}")
        print()

    # Save the bigram phrases to a CSV file
    with open('bigram_model.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Phrase', 'Score'])
        
        # Save each phrase and its score to CSV
        for phrase, score in bigram_model.phrasegrams.items():
            phrase_str = '_'.join([word.decode('utf-8') if isinstance(word, bytes) else word for word in phrase])
            writer.writerow([phrase_str, score])
    
    print("\nBigram model saved successfully as 'bigram_model.csv'.")
    return tokenized_reviews

# Bigram frequency analysis 
def count_bigram_frequencies(tokenized_reviews):
    # Helper function to generate bigrams from a list of words
    def generate_bigrams(words):
        return list(zip(words, words[1:]))

    # Flatten the list of bigrams across all reviews
    all_bigrams = [bigram for review in tokenized_reviews for bigram in generate_bigrams(review)]
    
    # Count occurrences
    bigram_counts = Counter(all_bigrams)
    
    # Filter bigrams with frequency > 3
    filtered_bigrams = {bigram: freq for bigram, freq in bigram_counts.items() if freq >= 3}
    
    # Get the top 10 most frequent bigrams from the filtered list
    top_10_bigrams = Counter(filtered_bigrams).most_common(10)

    # Print results
    print("\nTop 10 Most Frequent Bigrams:")
    for idx, ((word1, word2), freq) in enumerate(top_10_bigrams, start=1):
        print(f"{idx}. {word1}_{word2}: {freq} occurrences")

    # Save results to CSV
    with open('bigram_frequencies.csv', mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Bigram', 'Frequency'])
        for (word1, word2), freq in filtered_bigrams.items():
            writer.writerow([f"{word1}_{word2}", freq])

    print("\nBigram frequencies saved to 'bigram_frequencies.csv'.")
    
    # Visualization of top 10 bigrams
    bigrams, frequencies = zip(*top_10_bigrams)  # Unpacking tuples
    bigram_labels = [f"{word1} {word2}" for word1, word2 in bigrams]  # Formatting bigrams

    plt.figure(figsize=(12, 6))
    plt.barh(bigram_labels, frequencies, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Bigram')
    plt.title('Top 10 Most Frequent Bigrams')
    plt.gca().invert_yaxis()  
    plt.show()
    
# Define root causes under each health aspect using exact bigrams    
def Root_Cause_Bigrams(tokenized_reviews, bigram_df, topn=5):
    predefined_health_root_causes = {
        'Mental Health': {
            'Flight Delay Problems': ['delayed_flight', 'flight_delay', 'flight_delayed_hour', 'late_departure', 'missed_connection'],
            'Uncomfortable Flight': ['uncomfortable_flight', 'long_flight', 'long_time', 'cabin_temperature', 'poor_air'], 
            'Negative Experience': ['poor_experience', 'bad_experience', 'disappointing_experience'],
            'Noise Disturbance' : ['loud_passengers', 'crying_baby', 'engine_noise', 'noisy_cabin'],
        },
        'Physical Body Health': {
            'Uncomfortable Seat': ['uncomfortable_seat', 'seat_uncomfortable', 'hard_seat', 'seat_painful'],
            'Cramped Legroom': ['seat_cramped', 'cramped_seat', 'tight_space', 'no_legroom'],
            'Inadequate Sleeping Conditions': ['seat_recline', 'broken_recline', 'no_pillow','sleep_disrupted'],
            'Limited Movement': ['narrow_seat', 'aisle_blocked', 'restricted_movement', 'seatbelt_sign', 'long_sitting'],
        },
        'Emotional Health': {
            'Poor Service': ['poor_service', 'service_poor', 'service_slow', 'unresponsive_crew', 'ignored_request'],
            'Rude Staff': ['staff_rude', 'rude_attendant', 'impolite_crew', 'hostile_behavior'],
            'Lack of Empathy': ['no_apology', 'no_explanation', 'staff_uncaring', 'robotic_response'],
            'Boarding Chaos': ['boarding_mess', 'boarding_confusion', 'unclear_instructions', 'late_boarding'],
        },
        'Dietary Health': {
            'Bad Food Quality': ['food_poor', 'poor_food', 'bad_meal', 'cold_food', 'tasteless_meal'],
            'Limited Options': ['limited_food', 'no_vegetarian', 'no_meal_choice', 'meal_unavailable'],
            'Meal Timing Issues': ['late_meal', 'missed_meal', 'no_snack', 'meal_delayed'],
            'Unhygienic Food': ['dirty_utensils', 'spoiled_food', 'unclean_tray', 'unhygienic_conditions']
        }
    }
    
    print("Training Word2Vec model...")
    model = Word2Vec(sentences=tokenized_reviews, vector_size=100, window=5, min_count=2, workers=4)

    def get_bigram_vector(bigram):
        parts = bigram.split('_')
        if len(parts) != 2:
            return None
        word1, word2 = parts
        if word1 in model.wv and word2 in model.wv:
            return (model.wv[word1] + model.wv[word2]) / 2
        return None

    def find_similar_bigrams(seed_bigram):
        seed_vec = get_bigram_vector(seed_bigram)
        if seed_vec is None:
            return []

        similarities = []
        for bigram in bigram_df['Bigram'].unique():
            vec = get_bigram_vector(bigram)
            if vec is not None:
                sim = cosine_similarity([seed_vec], [vec])[0][0]
                similarities.append((bigram, sim))

        sorted_bigrams = sorted(similarities, key=lambda x: -x[1])
        return [bigram for bigram, sim in sorted_bigrams[:topn]]

    expanded_root_causes = {}

    for main_category, sub_dict in predefined_health_root_causes.items():
        expanded_root_causes[main_category] = {}
        for sub_category, seed_bigrams in sub_dict.items():
            expanded_root_causes[main_category][sub_category] = []
            for seed in seed_bigrams:
                similar = find_similar_bigrams(seed)
                expanded_root_causes[main_category][sub_category].extend(similar)

    return expanded_root_causes

# ============================
# Function 3: Topic Modeling With LDA & Wordcloud
# ============================

def build_lda_model(tokenized_reviews, num_topics=5):
    print("\n====== Processing LDA for All Reviews ======")

    # Create BoW model for all classes
    print("\nCreating Bag-of-Words (BoW) model for all reviews...")
    dictionary = corpora.Dictionary(tokenized_reviews)

    # Remove very common and rare words
    dictionary.filter_extremes(no_below=5, no_above=0.7, keep_n=10000)
    corpus = [dictionary.doc2bow(text) for text in tokenized_reviews]
    print("BoW model created.")

    # Train LDA model
    print(f"Training LDA model with {num_topics} topics...")
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=30, alpha='auto', eta='auto')
    print("LDA training completed.")

    # Display top words in each topic
    print("\nTop words in each topic:")
    for idx, topic in lda_model.print_topics(num_words=10):
        print(f"Topic {idx + 1}: {topic}")

    # Generate LDA visualization using pyLDAvis
    print("\nGenerating LDA visualization...")
    lda_display = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_display, "lda_visualization.html")
    print("LDA visualization saved as 'lda_visualization.html'")

    # Optional: Generate WordCloud for each topic
    generate_wordclouds(lda_model, dictionary, num_topics)

    return lda_model, dictionary, corpus

def generate_wordclouds(lda_model, dictionary, num_topics):
    for t in range(num_topics):
        plt.figure(figsize=(10, 8))
        word_freq = dict(lda_model.show_topic(t, 50))
        wordcloud = WordCloud(width=800, height=600, background_color="white").generate_from_frequencies(word_freq)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(f"wordcloud_topic_{t + 1}.png")
        plt.close()
    print(f"WordClouds topics saved successfully!")

# ============================
# Function 4: Sentiment And Emotion Analysis 
# ============================
# Load sentiment and emotion models  (VADER & DistilBert)
sia = SentimentIntensityAnalyzer()
emotion_classifier = pipeline(
    'sentiment-analysis', 
    model='bhadresh-savani/distilbert-base-uncased-emotion',
    truncation=True,     
    max_length=512         
)

# Define health-related keywords focused on passenger well-being
health_keyword_dict = {
    'service_and_staff': [
        'service', 'crew', 'staff', 'experience', 'cabin_crew', 'friendly', 'attentive', 
        'polite', 'helpful', 'rude', 'unprofessional', 'courteous', 'welcoming', 
        'support', 'interaction', 'communication', 'respectful', 'stress', 'anxiety', 
        'patience', 'agitation', 'comforting', 'reassurance', 'inconsistent', 'calm', 
        'emotional_support', 'wellbeing', 'customer_service', 'hospitality', 
        'frustration', 'anger', 'emotional_health', 'aggravation', 'annoyance'
    ],
    'flight_experience': [
        'time', 'check', 'boarding', 'passenger', 'lounge', 'delay', 'timing', 
        'departure', 'arrival', 'security', 'queue', 'baggage', 'priority', 
        'transfer', 'connection', 'immigration', 'gate', 'boarding_process', 
        'smooth', 'chaotic', 'process', 'efficiency', 'stress', 'anxiety', 
        'long_wait', 'rushed', 'tiredness', 'fatigue', 'discomfort', 
        'exhaustion', 'jetlag', 'sleep_deprivation', 'frustration', 'mental_health'
    ],
    'food_and_drink': [
        'food', 'meal', 'choice', 'drink', 'beverage', 'menu', 'snack', 
        'options', 'quality', 'catering', 'breakfast', 'lunch', 'dinner', 
        'portion', 'variety', 'freshness', 'dietary', 'nutrition', 'hydration', 
        'allergic', 'intolerance', 'sick', 'stomach', 'digestive', 'upset', 
        'hygiene', 'healthy', 'unhygienic', 'vomiting', 'nausea', 'contamination', 
        'diarrhea', 'malnutrition', 'food_poisoning'
    ],
    'seat_comfort': [
        'seat', 'comfort', 'business_class', 'aircraft', 'legroom', 'recline', 
        'cramped', 'position', 'space', 'cushion', 'headrest', 'armrest', 
        'seat_width', 'seat_pitch', 'footrest', 'adjustable', 'aisle', 
        'window_seat', 'economy_class', 'premium_economy', 'sleep_quality', 
        'back_pain', 'body_ache', 'discomfort', 'posture', 'stiffness', 
        'fatigue', 'numbness', 'strain', 'pressure', 'circulation', 'muscle_pain', 
        'pain', 'physical_health', 'pressure_point', 'swelling'
    ]
}

# Sentiment analysis function (VADER)
def get_sentiment(text):
    """Get sentiment polarity using VADER."""
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def detect_emotion(text):
    result = emotion_classifier(text) 
    return result[0]['label']

# Identify aspects and assign sentiment
def identify_aspects(text, aspect_dict):
    """Match keywords to relevant health categories."""
    aspects_found = []
    for category, keywords in aspect_dict.items():
        for word in keywords:
            if word in text:
                aspects_found.append(category)
                break
    return aspects_found

def analyze_review(text, aspect_dict):
    """Analyze review for aspects and sentiment with aggregation."""
    aspects = identify_aspects(text, aspect_dict)
    sentiment = get_sentiment(text)
    emotion = detect_emotion(text)

    # Store summarized results per review
    aspect_summary = {}

    for aspect in aspects:
        aspect_summary[aspect] = {
            'sentiment': sentiment,
            'emotion': emotion,
            'sentence': text 
        }

    return aspect_summary

# ============================
# Function 5: Map Aspect Sentiment To Health 
# ============================
def map_aspect_to_health(aspect_sentiment):
    
    """Map aspect sentiment to relevant health categories with sentence extraction."""
    health_impact_mapping = {
        'service_and_staff': 'emotional_health',
        'flight_experience': 'mental_health',
        'food_and_drink': 'dietary_health',
        'seat_comfort': 'physical_body_health',
    }

    # Define default result when no aspect is detected
    default_result = {
        'emotional_health': {'impact': 'neutral', 'emotion': 'neutral', 'sentence': ''},
        'mental_health': {'impact': 'neutral', 'emotion': 'neutral', 'sentence': ''},
        'dietary_health': {'impact': 'neutral', 'emotion': 'neutral', 'sentence': ''},
        'physical_body_health': {'impact': 'neutral', 'emotion': 'neutral', 'sentence': ''},
    }

    health_results = default_result.copy()

    for aspect, data in aspect_sentiment.items():
        # Check if the aspect exists in the mapping
        if aspect in health_impact_mapping:
            category = health_impact_mapping[aspect]
            sentiment = data['sentiment']
            emotion = data['emotion']
            sentence = data['sentence']

            # Determine impact based on sentiment
            impact = 'positive' if sentiment == 'positive' else 'negative' if sentiment == 'negative' else 'neutral'
            
            # Update the health category with relevant data
            health_results[category] = {
                'impact': impact,
                'emotion': emotion,
                'sentence': sentence
            }
    
    # Flatten results to maintain one row per review
    flattened_results = []
    for category, result in health_results.items():
        flattened_results.append({
            'health_category': category,
            'impact': result['impact'],
            'emotion': result['emotion'],
            'sentence': result['sentence']
        })
    
    return flattened_results

# ============================
# Function 6: Save Health Impact Results
# ============================
def save_results_to_csv(aspect_results, all_health_results):
    """Save aspect and health results to CSV."""
    # Save one row per review for health results
    health_df = pd.DataFrame(all_health_results)
    cols = ['date'] + [col for col in health_df.columns if col != 'date']
    health_df = health_df[cols]
    health_df.to_csv('health_impact_results.csv', index=False)
    print("Health impact results saved as 'health_impact_results.csv'.")
    
# ============================
# Function 7: Data Visualization
# ============================
def plot_health_impact(health_results):
    df = pd.DataFrame(health_results)
    
    if not df.empty:
        df['impact'] = df['impact'].replace({'positive': 'good', 'negative': 'bad'})

        # Figure 1 :Plot Health Impact (Impact vs Health Category)
        health_summary = df.groupby(['health_category', 'impact']).size().reset_index(name='count')
        impact_order = ['bad', 'neutral', 'good']

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            data=health_summary,
            x='health_category',        
            y='count',
            hue='impact',
            hue_order=impact_order,
            palette={'good': '#4CAF50', 'neutral': '#FFC107', 'bad': '#F44336'}
        )
        plt.title('Sentiment Analysis of Health-Related Aspects in Airline Reviews')
        plt.xlabel('Health Category')
        plt.ylabel('Number of Mentions')

        # Add labels on each bar
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}',               
                            (p.get_x() + p.get_width() / 2., p.get_height()),  
                            ha='center', va='center',               
                            xytext=(0, 8),                         
                            textcoords='offset points',
                            fontsize=10, color='black')              
        plt.show()
    

        # Figure 2: Plot Emotion Distribution by Airline Factors
        reverse_mapping = {
        'emotional_health': 'service_and_staff',
        'mental_health': 'flight_experience',
        'dietary_health': 'food_and_drink',
        'physical_body_health': 'seat_comfort',
        }

        # Apply the mapping to rename health categories back to airline factors
        df['airline_factor'] = df['health_category'].map(reverse_mapping)

        # Group by airline factor and emotion
        emotion_summary = df.groupby(['airline_factor', 'emotion']).size().reset_index(name='count')

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(  # Assign sns.barplot to ax
            data=emotion_summary,
            x='airline_factor',
            y='count',
            hue='emotion',
            palette='coolwarm'
        )

        plt.title('Emotion Distribution Across Airline Service Factors')
        plt.xlabel('Airline Factors')
        plt.ylabel('Number of Mentions')
        plt.legend(title='Emotion')

        # Add labels on each bar
        for p in ax.patches:
            if p.get_height() > 0:  # Avoid showing "0" on empty bars
                ax.annotate(f'{int(p.get_height())}',               
                    (p.get_x() + p.get_width() / 2., p.get_height()),  
                    ha='center', va='bottom',               
                    xytext=(0, 8),  # Offset text slightly above the bar                         
                    textcoords='offset points',
                    fontsize=10, color='black')

        plt.show()
        
def plot_root_cause_by_health_aspect():
    # Rule-Based Analysis
    df = pd.read_csv('bigram_frequencies.csv')

    # Define root causes under each health aspect using exact bigrams
    health_root_causes = {
        'Mental Health': {
            'Flight Delay Problems': ['delayed_flight', 'delay_flight', 'flight_delayed', 'flight_delay', 'flight_delayed_hour',
                                      'flight_late', 'late_departure', 'delayed_departure'],
            'Uncomfortable Flight': ['uncomfortable_flight', 'long_flight', 'long_time', 'pretty_poor', 'quality_poor'], 
            'Negative Experience': ['poor_experience', 'bad_experience', 'disappointing_experience', 'overall_poor'],
        },
        'Physical Body Health': {
            'Uncomfortable Seat': ['uncomfortable_seat', 'seat_uncomfortable', 'seat_little', 'seat_hard', 'hard_seat'],
            'Cramped Legroom': ['seat_cramped', 'cramped_seat', 'little_space'],
            'Inadequate Sleeping Conditions': ['seat_recline', 'recline_seat'],
            'Limited Movement': ['narrow_seat', 'seat_narrow'],
        },
        'Emotional Health': {
            'Poor Service': ['poor_service', 'service_poor', 'service_slow', 'never_came'],
            'Rude Staff': ['staff_rude', 'rude_staff'],
            'Boarding Chaos': ['late_boarding'],
        },
        'Dietary Health': {
            'Bad Food Quality': ['food_poor', 'poor_food'],
            'Limited Options': ['limited_food', 'lack_food'],
        }
    }

    # Set the seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    custom_palette = sns.color_palette("pastel")

    for health_aspect, causes in health_root_causes.items():
        root_cause_freqs = []

        for cause_name, bigrams in causes.items():
            total_freq = df[df['Bigram'].isin(bigrams)]['Frequency'].sum()
            root_cause_freqs.append((cause_name, total_freq))

        cause_df = pd.DataFrame(root_cause_freqs, columns=['Root Cause', 'Frequency'])
        cause_df = cause_df.sort_values(by='Frequency', ascending=False).head(3)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=cause_df, x='Root Cause', y='Frequency',
                         palette=custom_palette, edgecolor='black')

        plt.title(f"Top Root Causes Affecting {health_aspect}", fontsize=16, weight='bold')
        plt.ylabel("Total Frequency", fontsize=12)
        plt.xlabel("")
        plt.xticks(rotation=15, ha='center', fontsize=11)

        # Bar labels
        for bar in ax.patches:
            height = int(bar.get_height())
            ax.annotate(f'{height}',
                        (bar.get_x() + bar.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='semibold',
                        color='black', xytext=(0, 5),
                        textcoords='offset points')

        ax.spines[['top', 'right']].set_visible(False)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()
        
def plot_yearly_sentiment():
    # Load your CSV data
    file_path = 'health_impact_results.csv'  # Replace with your actual file path
    df = pd.read_csv(file_path)

    # Convert date to datetime and extract year
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['year'] = df['date'].dt.year

    # Clean and filter
    df = df.dropna(subset=['year'])
    
    # Remap the sentiment columns
    sentiment_mapping = {
        'positive': 'good',
        'neutral': 'neutral',
        'negative': 'bad'
    }
    df['impact'] = df['impact'].map(sentiment_mapping)

    # Filter out any rows with invalid or unmapped impacts
    df = df[df['impact'].isin(['good', 'neutral', 'bad'])]

    # Group by health category, year, and impact
    grouped = df.groupby(['health_category', 'year', 'impact']).size().reset_index(name='count')

    # Get list of health categories
    health_categories = grouped['health_category'].unique()

    # Plotting line graph per health category
    for category in health_categories:
        category_data = grouped[grouped['health_category'] == category]

        # Pivot so each sentiment becomes a column
        pivot_df = category_data.pivot(index='year', columns='impact', values='count').fillna(0)

        # Plot
        plt.figure(figsize=(8, 5))
        for impact in ['good', 'neutral', 'bad']:
            if impact in pivot_df.columns:
                plt.plot(pivot_df.index, pivot_df[impact], marker='o', label=impact.replace('_', ' ').title())

        plt.title(f"Yearly Passenger Sentiments - {category.replace('_', ' ').title()}")
        plt.xlabel("Year")
        plt.ylabel("Number of Reviews")
        plt.legend(title='Health Impact')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
# ============================
# STEP 7: MAIN EXECUTION
# ============================
def main():
    # Function 1 (Data Loading and Preprocessing)
    file_path = 'British_Airline_Dataset.csv'
    df = load_reviews(file_path)

    # Function 2 (Bigram Detection and Frequency Analysis)
    tokenized_reviews = list(df['cleaned_tokens'])
    tokenized_reviews = build_phrases(tokenized_reviews)
    count_bigram_frequencies(tokenized_reviews)
    
    bigram_df = pd.read_csv('bigram_frequencies.csv')
    expanded_causes = Root_Cause_Bigrams(tokenized_reviews, bigram_df, topn=5)
    for category, bigrams in expanded_causes.items():
        print(f"\n{category}:")
        print(bigrams)
    
    # Function 3 (Topic Modeling With LDA & Wordcloud)
    build_lda_model(tokenized_reviews)

    # Function 4 (Sentiment And Emotion Analysis) 
    print("Performing Sentiment and Emotion Analysis...")
    df['aspect_sentiment'] = df['content'].apply(lambda x: analyze_review(x, health_keyword_dict))
    df.to_csv('sentiment_results.csv', index=False, encoding='utf-8')

    # Function 5 (Map Aspect Sentiment To Health)
    print("Mapping in progress...")
    all_health_results = []
    for date, aspect_sentiment in zip(df['date'], df['aspect_sentiment']):
        results = map_aspect_to_health(aspect_sentiment)
        for res in results:
            res['date'] = date 
            all_health_results.append(res)

    # Function 6 (Save Health Impact Results)
    save_results_to_csv(df['aspect_sentiment'], all_health_results)
    
    # Function 7 (Data Visualization For Airline Factors)
    plot_health_impact(all_health_results)
    plot_root_cause_by_health_aspect()
    plot_yearly_sentiment()
    

# ============================
#          RUN SCRIPT
# ============================
if __name__ == "__main__":
    main()