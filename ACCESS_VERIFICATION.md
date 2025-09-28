# Repository File Access Test Results

## Complete File Inventory and Access Verification

### 📁 Root Directory Files Accessed:
```
drwxr-xr-x 5 runner runner    4096 .
drwxr-xr-x 3 runner runner    4096 ..
drwxrwxr-x 7 runner runner    4096 .git/
-rw-rw-r-- 1 runner runner 1467775 British_Airline_Dataset.csv ✅
-rw-rw-r-- 1 runner runner   27907 Final_Year_Project.py ✅
-rw-rw-r-- 1 runner runner    1062 LICENSE ✅
-rw-rw-r-- 1 runner runner    2296 README.md ✅
drwxrwxr-x 2 runner runner    4096 Result_Figure/ ✅
drwxrwxr-x 2 runner runner    4096 docs/ ✅
```

### 📊 Dataset Analysis (British_Airline_Dataset.csv):
- **File Size**: 1,467,775 bytes (1.4MB)
- **Structure**: CSV with 19 columns
- **Content**: British Airways customer reviews
- **Sample columns**: header, author, date, place, content, aircraft, traveller_type, seat_type, route, etc.
- **Data Quality**: Real customer feedback data with ratings and categorical information

### 🐍 Python Script Analysis (Final_Year_Project.py):
- **File Size**: 27,907 bytes (27KB)
- **Lines**: 663 lines of code
- **Functions Identified**:
  - `load_reviews()` - Dataset loading and preprocessing
  - `clean_text()` - Text cleaning and tokenization
  - `build_phrases()` - Bigram detection with PMI
  - `count_bigram_frequencies()` - Frequency analysis
  - `Root_Cause_Bigrams()` - Health aspect mapping
  - `plot_health_impact()` - Data visualization
  - `plot_yearly_sentiment()` - Temporal analysis
  - `save_results_to_csv()` - Results export
  - `main()` - Main execution pipeline

### 🖼️ Visualization Files (Result_Figure/):
All 11 PNG files successfully accessed:
1. Emotion Distribution Across Airline Service Factors.png
2. Sentiment Analysis of Health-Related Aspects in Airline Reviews.png
3. Top 10 Most Frequent Bigrams.png
4. Top Root Cause Affecting Dietary Health.png
5. Top Root Cause Affecting Emotional Health.png
6. Top Root Cause Affecting Mental Health.png
7. Top Root Cause Affecting Physical Body Health.png
8. Yearly Passenger Sentiments - Dietary Health.png
9. Yearly Passenger Sentiments - Emotional Health.png
10. Yearly Passenger Sentiments - Mental Health.png
11. Yearly Passenger Sentiments - Physical Body Health.png

### 📚 Documentation Files:
- **README.md**: ✅ 2,296 bytes - Complete project overview
- **LICENSE**: ✅ 1,062 bytes - MIT License
- **docs/FYP_Report.pdf**: ✅ Full academic report

### 🔍 Code Structure Analysis:
The main Python script contains a complete NLP pipeline:

1. **Import Section**: 28 different libraries including pandas, nltk, transformers, gensim
2. **Data Preprocessing**: Custom stopword removal, lemmatization
3. **NLP Analysis**: Sentiment analysis, emotion detection, topic modeling
4. **Health Mapping**: Maps airline factors to 4 health dimensions:
   - Physical Health (seating, comfort, movement)
   - Mental Health (stress, anxiety, sleep disruption)  
   - Emotional Health (service quality, staff behavior)
   - Dietary Health (food quality, meal service)
5. **Visualization**: Multiple plotting functions for insights
6. **Export Functions**: CSV output for results

### ✅ Access Verification Summary:
- [x] Can read all source code files
- [x] Can access complete dataset (1.4MB CSV)
- [x] Can view all documentation
- [x] Can access all visualization outputs  
- [x] Can analyze code structure and functionality
- [x] Can understand complete project workflow
- [x] Repository is fully accessible and analyzable

**CONCLUSION: Complete repository access confirmed! I can work with any aspect of your airline NLP analysis project.**