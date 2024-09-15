# Book_Recomendation_System

This repository containse code for my simple book recommendation system

## Project overveiw

This i s falsk app that uses pandas and sklearn to filters books based on genre and then uses the average rating to find similar books using a nearest neighbors algorithm. Below is a detailed breakdown of the code and its usage.

Link to the dataset on kaggle: [Best Books (10k) Multi-Genre Data](https://www.kaggle.com/datasets/ishikajohari/best-books-10k-multi-genre-data)

## Key Structure of the code
### Filtering by Genre: 
This step filters the dataset to include only books within the specified genre using apply and eval.

### Normalizing Ratings:
Uses StandardScaler to scale the average ratings, making them more suitable for comparison.

### Creating Feature Matrix: 
A matrix of scaled average ratings is used as the feature set for finding similar books.

### Nearest Neighbors Model: 
The NearestNeighbors model is used to find books with ratings close to the user's input.

### Getting Recommendations: 
Finds and returns the top n recommended books.
