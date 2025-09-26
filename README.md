# Movie Recommendation System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AhmedEhab2022/movie_recommendation_system/blob/main/Movie_Recommendation_System.ipynb)

A comprehensive machine learning project that implements multiple collaborative filtering approaches to build an intelligent movie recommendation system. This project demonstrates three distinct recommendation algorithms and evaluates their performance using the MovieLens 100K dataset to provide personalized movie suggestions.

## Project Overview

This repository contains a complete movie recommendation workflow that analyzes user-movie interactions to predict user preferences and generate personalized movie recommendations. The system implements user-based collaborative filtering, item-based collaborative filtering, and matrix factorization techniques to deliver accurate recommendations.

## Dataset

**Dataset**: MovieLens 100K Dataset from Kaggle  
**Link**: https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset

The project uses the MovieLens 100K dataset containing:

- **User Ratings**: 100,000 ratings from 943 users on 1,682 movies
- **Rating Scale**: 1-5 stars
- **Movie Information**: Titles, genres, release dates
- **User Demographics**: Age, gender, occupation
- **Time Period**: Ratings collected over 7 months

The dataset provides rich interaction data with no missing values in the core rating matrix.

## Methodology

### Data Preprocessing

- Exploratory data analysis with comprehensive visualizations
- Data cleaning and duplicate removal
- User-item matrix construction for collaborative filtering
- Train-test split (80-20) for model evaluation
- Rating distribution analysis and statistical insights

### Recommendation Algorithms

1. **User-Based Collaborative Filtering (UBCF)**

   - Cosine similarity computation between users
   - Neighbor-based rating prediction
   - Top-K similar user identification

2. **Item-Based Collaborative Filtering (IBCF)**

   - Movie-to-movie similarity matrix generation
   - Item neighborhood formation
   - Rating prediction based on similar movies

3. **Matrix Factorization (SVD)**
   - Singular Value Decomposition implementation
   - Latent feature extraction (10-100 components)
   - Dimensionality reduction for scalability
   - Hyperparameter tuning for optimal performance

## Key Findings

The analysis reveals **performance comparison across three algorithms**:

| Algorithm                      | Approach                    | Strengths                          | Use Cases                  | Precision@5 |
| ------------------------------ | --------------------------- | ---------------------------------- | -------------------------- | ----------- |
| **User-Based Collaborative**   | Find similar users          | Captures user preferences patterns | New item recommendations   | ~6%         |
| **Item-Based Collaborative**   | Find similar movies         | Stable, interpretable results      | Content-based suggestions  | ~5%         |
| **Matrix Factorization (SVD)** | Latent factor decomposition | Scalable, handles sparsity well    | Large-scale recommendation | ~33%        |

## Technologies Used

- **Python Libraries**: pandas, numpy, matplotlib, seaborn
- **Machine Learning**: scikit-learn (TruncatedSVD, cosine_similarity, train_test_split)
- **Mathematical Operations**: Matrix operations, similarity computations
- **Evaluation Metrics**: Precision@K for recommendation quality assessment
- **Environment**: Colab

## Requirements

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Upload the dataset to your Google Drive in the appropriate folder
3. Run all cells in sequence

### Option 2: Local Setup

1. Clone this repository: `git clone https://github.com/AhmedEhab2022/movie_recommendation_system.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Open `Movie_Recommendation_System.ipynb` in Jupyter
4. Update the data path to point to your local dataset
5. Run all cells

## Project Structure

```
movie-recommendation-system/
├── Movie_Recommendation_System.ipynb   # Main analysis notebook
├── ml-100k/                            # MovieLens dataset directory
│   ├── u.data                          # User ratings data
│   ├── u.item                          # Movie information
│   ├── u.user                          # User demographics
│   └── README                          # Dataset documentation
├── README.md                           # Project documentation
├── LICENSE                             # License information
└── requirements.txt                    # Python dependencies
```

## Key Visualizations

The project includes comprehensive visualizations:

- **Rating Distribution**: Histogram showing user rating patterns
- **User-Item Matrix**: Sparse matrix visualization of user-movie interactions
- **Similarity Heatmaps**: User and item similarity matrices
- **Recommendation Results**: Top-N movie suggestions for sample users
- **Performance Metrics**: Precision@K evaluation across algorithms

## Business Applications

- **Streaming Platforms**: Netflix, Amazon Prime, Hulu recommendation engines
- **E-commerce**: Product recommendation systems
- **Content Discovery**: Help users find relevant movies based on preferences
- **User Engagement**: Increase platform usage through personalized content
- **Marketing Strategy**: Targeted movie promotions and campaigns

## Analysis Highlights

- **Algorithm Comparison**: Comprehensive evaluation of three recommendation approaches
- **Hyperparameter Tuning**: SVD component optimization (10-100 components)
- **Evaluation Framework**: Precision@K metric implementation
- **Scalability Analysis**: Performance assessment on 100K rating dataset
- **Cold Start Problem**: Analysis of recommendation quality for new users

## Key Insights

1. **Matrix Factorization (SVD)** provides scalable solutions for large datasets
2. **Item-based filtering** offers more stable and interpretable recommendations
3. **User-based filtering** captures dynamic user preference patterns effectively
4. **Hyperparameter tuning** significantly improves recommendation quality
5. **Evaluation metrics** are crucial for comparing algorithm performance objectively

## Performance Evaluation

The system uses **Precision@K (P@K)** evaluation technique:

- **Precision@5**: Measures relevant recommendations in the top 5 suggestions
- **Cross-validation**: Train-test split ensures unbiased evaluation
- **User-specific metrics**: Individual precision scores for comprehensive assessment
- **Algorithm comparison**: Direct performance comparison across methods

## Future Enhancements

- **Deep Learning Integration**: Neural collaborative filtering implementation
- **Hybrid Approaches**: Combining multiple recommendation techniques
- **Real-time Recommendations**: Online learning and streaming data processing
- **Content-based Features**: Integration of movie genres and metadata
- **Advanced Metrics**: Implementation of Recall, F1-score, and NDCG metrics

## Repository Link

**GitHub**: https://github.com/AhmedEhab2022/movie_recommendation_system.git

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

