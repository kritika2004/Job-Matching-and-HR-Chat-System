# Job-Matching-and-HR-Chat-System

Project Overview
This project comprises two main components:

Job Matching Script: Automates the process of matching resumes to job descriptions using various similarity measures (TF-IDF, Jaccard, BERT, Word2Vec, etc.). Provides insights into the relevance of resumes to job postings using visualization and interactive analysis.

Chat Job for HR: A conversational AI tool designed to assist HR personnel by providing quick insights, answering queries about job matches, and managing job-related conversations.

Part 1: Job Matching Script
Description
This script analyzes and ranks resumes against job postings using advanced natural language processing (NLP) techniques. Multiple similarity measures are implemented to compare resumes with job descriptions. Visualizations, interactive plots, and tabular results are provided to help HR teams quickly identify the best candidates.

Features
Data Cleaning and Preprocessing:

Normalizes text (lowercasing, removing special characters, etc.).
Handles missing values to ensure clean and accurate results.
Similarity Calculation Methods:

TF-IDF: Measures cosine similarity between term-weighted vectors.
Jaccard Similarity: Calculates overlap of unique terms between resumes and jobs.
BERT + FAISS: Leverages contextual embeddings from the BERT model and accelerates similarity search using FAISS.
Word2Vec: Captures semantic similarities using word embeddings.
Visualization and Results Display:

Heatmaps for similarity comparisons.
Bar plots for top terms in resumes and job descriptions.
Interactive scatter plots to explore relationships between resumes and job matches.
Tabular summaries of top matches for each similarity method.
Interactive and Detailed Insights:

Displays top matches and their similarity scores for a random subset of resumes.
Aggregates results from different methods for better decision-making.
Prerequisites
Libraries Required:

pandas, numpy, matplotlib, seaborn, plotly, nltk, faiss, gensim
Hugging Face Transformers (transformers), sklearn
Install missing packages using:
bash
Copy code
pip install pandas numpy matplotlib seaborn plotly transformers sklearn faiss gensim
Data Input:

Resumes (Res.csv): A CSV file containing resumes with an "ID" and "Resume_str" column.
Job Postings (pos.csv): A CSV file containing job descriptions with a "title" and "description" column.
Hardware:

GPU support (optional but recommended for BERT processing).
How to Use
Load Data:

Upload Res.csv and pos.csv in the appropriate format.
Run Similarity Calculations:

Choose a similarity method (e.g., TF-IDF, Jaccard).
Execute the corresponding function to compute similarity scores and display results.
Visualize Results:

Use the heatmaps, bar charts, and scatter plots to analyze matches.
Explore top terms and similarities interactively.
Analyze Results:

Review tabular summaries of the top 3 matches for each resume.
Use scatter plots to compare the strength of similarity across different job postings.
Outputs
Similarity Results: Ranked job matches for each resume based on chosen methods.
Visualizations:
Heatmaps, bar plots, and scatter plots.
Interactive Analysis: Drill down into top matches and explore candidate-job relationships.
Part 2: Chat Job for HR
Description
The HR Chat module leverages conversational AI to streamline communication between HR personnel and job candidates. It allows HR to interact with the system to retrieve job matching insights, manage conversations, and provide a seamless experience for candidate screening.

Features
Real-Time Interaction:

HR can chat with the system to:
Query top matches for a specific candidate.
Retrieve job posting details.
Get similarity scores and insights dynamically.
Advanced Conversational AI:

Built on top of pretrained language models for contextual understanding.
Responds to HR queries with relevant job data and recommendations.
Integration with Job Matching Script:

Uses the results from Part 1 to provide meaningful answers to HR queries.
Ensures consistency between analytical outputs and conversational responses.
Ease of Use:

Simple text-based chat interface.
Optimized for quick responses and minimal setup.
Prerequisites
Libraries Required:

Hugging Face Transformers (transformers)
flask or similar web framework (if deployed as a standalone system).
Data Requirements:

Processed results from the Job Matching Script.
How to Use
Run the Chat Interface:

Start the conversational AI system using:
bash
Copy code
python hr_chat_system.py
Open the chat interface in a web browser or terminal.
Ask Queries:

Example queries:
"Show me the top matches for Resume ID 12345."
"Which jobs are best suited for this candidate?"
"Provide similarity scores for all positions related to 'Data Scientist'."
Integrate with Job Matching Results:

Ensure that the output files from Part 1 are accessible to the chat system.
Outputs
Dynamic Responses: Contextual answers to HR queries.
Data Insights: Interactive and personalized recommendations for hiring decisions.
Conclusion
This two-part project aims to simplify the hiring process by combining advanced NLP techniques for job matching with an intuitive conversational interface for HR professionals. By automating and enhancing candidate-job matching, this tool provides valuable insights, reduces manual effort, and improves hiring efficiency.

