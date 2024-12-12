# Job-Matching-and-Job_Seekers-Chat-System

## Project Overview
This project comprises two main components:

Job Matching Script: Automates the process of matching resumes to job descriptions using various similarity measures (TF-IDF, Jaccard, BERT, Word2Vec). Provides insights into the relevance of resumes to job postings using visualization and interactive analysis.

Chat Bot for Job Seekers: A conversational AI tool designed to assist job seekers by providing quick insights, answering queries about job matches, and managing job-related conversations.

## Part 1: Job Matching Script
Description
This script analyzes and ranks resumes against job postings using advanced natural language processing (NLP) techniques. Multiple similarity measures are implemented to compare resumes with job descriptions. Visualizations, interactive plots, and tabular results are provided to help job seekers quickly identify the most suitable jobs.

### Features
#### Data Cleaning and Preprocessing:

Normalizes text (lowercasing, removing special characters, etc.).
Handles missing values to ensure clean and accurate results.

#### Similarity Calculation Methods:

TF-IDF: Measures cosine similarity between term-weighted vectors.
Jaccard Similarity: Calculates the overlap of unique terms between resumes and jobs.
BERT + FAISS: Leverages contextual embeddings from the BERT model and accelerates similarity search using FAISS.
Word2Vec: Captures semantic similarities using word embeddings.

### Visualization and Results Display:

#### Heatmaps for similarity comparisons.
![image](https://github.com/user-attachments/assets/3aedd53c-f646-4c74-86d0-133c5192d65b)

![image](https://github.com/user-attachments/assets/92c9f98b-a8ab-4130-a5ae-aade2d5e8655)


#### Bar plots for top terms in resumes and job descriptions.
![image](https://github.com/user-attachments/assets/0e23029d-0df2-4d58-b269-6149c15f5f03)

#### Interactive scatter plots to explore relationships between resumes and job matches.

![image](https://github.com/user-attachments/assets/107d73bb-d707-40e0-9aa8-67e4e8ab220e)

![image](https://github.com/user-attachments/assets/86ade0b5-e34e-4a72-8570-7c8800c902da)


#### Tabular summaries of top matches for each similarity method.

![Screenshot (15)](https://github.com/user-attachments/assets/a56c3795-eb20-4fb9-9dde-47d0508c1d70)



### Prerequisites
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

### How to Use
Load Data:

Upload Res.csv and pos.csv in the appropriate format.
Run Similarity Calculations:

Choose a similarity method (e.g., TF-IDF, Jaccard).
Execute the corresponding function to compute similarity scores and display results.

### Key Takeaways
#### 1. TF-IDF Results
The TF-IDF algorithm appears to provide a meaningful ranking of resumes, highlighting roles such as "Administrative Assistant," "Marketing Coordinator," and "Project Architect."
Average Similarity scores are relatively low (ranging around 0.11–0.22), which is expected for sparse data like resumes.
TF-IDF is performing well in identifying matches where textual overlap (job titles and keywords) is higher.
#### 2. Jaccard Similarity Results
Jaccard scores are consistently low, with Average Similarity scores around 0.09–0.15. This reflects its limitation in comparing longer texts (e.g., resumes), as Jaccard is highly sensitive to small differences in word overlap.
Matches like "Administrative Assistant" and "Marketing Coordinator" appear frequently, indicating that resumes with these roles likely share a higher proportion of common words.
#### 3. BERT + FAISS Results
BERT embeddings with FAISS yield the most robust results, with Average Similarity scores in the 49–65 range, which is higher than other methods.
This method captures semantic relationships between resumes and roles, explaining why matches like "Administrative Coordinator" and "Marketing Coordinator" are consistent.
This algorithm is particularly effective in identifying resumes with related but not identical content, making it valuable for nuanced role matching.
#### 4. Word2Vec Results
The Word2Vec-based scores are notably high (around 0.99), suggesting that embeddings are overfitting on certain repetitive features or failing to differentiate subtle variations.
Matches like "SALES" and "Legal Secretary" dominate, which may not reflect meaningful semantic relationships but rather generic token similarities.
While Word2Vec can provide value, it appears less reliable for resume-to-job matching in this case.
### Recommended Algorithms
BERT + FAISS: The best-performing algorithm for capturing semantic and contextual relevance. It is ideal for finding resumes that align with nuanced job descriptions.
TF-IDF: Performs well in identifying straightforward textual overlaps and is computationally lightweight. Useful for bulk pre-screening.
Jaccard: Has limited utility due to its reliance on exact word overlap, which is less relevant for resume data.
Word2Vec: Its high scores likely indicate overgeneralization, making it less useful for distinguishing nuanced matches.


## Part 2: Chat Bot for Job Seekers
### Description
The ChatBot module leverages conversational AI to streamline communication between job seekers and HR. It allows job seekers to interact with the system to find relevant job openings, ask questions, and receive timely updates throughout the application process.

![image](https://github.com/user-attachments/assets/03a36db4-23b7-453f-8766-24c668ac10ea)

This section outlines how the fine-tuned conversational model (using GPT-Neo and a conversational dataset) enhances career advisory responses. This approach leverages a pre-trained model, fine-tuned on domain-specific data, to provide tailored and accurate suggestions.

### Conversational Dataset Design
The dataset is structured with questions and answers addressing diverse career-related topics, such as:
Data science job roles
Resume-building for IT and marketing
Tools and skills for specific professions
Certifications for construction and IT careers

### Why it works:

The conversational format aligns with real-world user queries.
It captures a range of questions that span technical and non-technical roles, improving generalization.
The explicit Q&A structure aids in building contextual understanding for diverse inquiries.

### Fine-Tuning Process
Using GPT-Neo (EleutherAI/gpt-neo-125M), the model was fine-tuned on the Q&A dataset with the following steps:

#### Dataset Preparation:
Combined the question and answer into a single text field for input-output mapping.
Tokenized with truncation and padding to ensure uniform input sequences.

#### Fine-Tuning Objectives:
Train the model to accurately map diverse questions to domain-specific answers.
Improve fluency, relevance, and consistency in generating career advice.

#### Training Strategy:
Optimized using Trainer from Hugging Face, with hyperparameters tailored for conversational tasks:
Batch size: 4 for efficient computation.
Epochs: 3 to balance accuracy and training time.
Learning rate: 5e-5 to refine model weights incrementally.
Evaluation on a 10% test split to monitor generalization.

#### Final Deployment:
Fine-tuned model and tokenizer saved for deployment.
Integrated with Gradio for a user-friendly interface, allowing real-time interaction.

###  Key Strengths of the Fine-Tuned Model
Domain-Specific Insights:
The model provides targeted advice for data science, IT, marketing, and other careers.
Fine-tuning enhances its ability to address niche queries like resume optimization and skill prioritization.

Improved Contextual Understanding:
By training on domain-specific Q&A, the model grasps the intent behind career-related questions.
For example, it differentiates between technical IT certifications (e.g., AWS) and general IT skills (e.g., troubleshooting).

Dynamic Interaction:
Gradio integration facilitates real-time question answering with adjustable parameters (e.g., response length, temperature).
It ensures the chatbot can handle diverse queries effectively.

### Opportunities for Further Optimization
Dataset Expansion:
Add more varied questions (e.g., specific industries, career paths for unique skills).
Include queries with ambiguous phrasing to test the model's contextual understanding.
Use RAG to utilize resume and job vector databases for more concise outputs.

Model Scaling:
Upgrade to larger versions of GPT-Neo or GPT-J for enhanced performance.
Explore newer architectures like LLaMA or Falcon for improved efficiency.

Interactive Features:
Integrate response feedback from users to fine-tune iteratively.
Add capabilities like generating sample resumes or recommending career resources.

Reducing overmatching:
TF-IDF or other vectorization methods to give higher weight to keywords (e.g., specific skills, job titles) to solve overmatching based on non-keywords such as conjunctions or other commonly used words.

Example queries:
"Show me the top matches for Resume ID 12345."
"Which jobs are best suited for this candidate?"
"Provide similarity scores for all positions related to 'Data Scientist'."
Integrate with Job Matching Results:


### This two-part project aims to simplify the hiring process by combining advanced NLP techniques for job matching with an intuitive conversational interface for Job seekers. By automating and enhancing candidate-job matching, this tool provides valuable insights, reduces manual effort, and improves hiring efficiency.

## Literature Review and Existing Research

# Risks
There are many risks we seeks to avoid when setting up machine learning systems for matching resumes with job postings.
the first risk widely covered in the literature is around bias in the algorithm.
Every pretrained model introduces the risk of bias in the training data. This is a particular risks in models such as BERT which depend heavily on their training data. There is eveidence of models favoring specific demographics or excluding non-traditional resumes from being selected for advancement in the stages of the hiring process. There are ways we can mitigate this risk of bias. First and foremost, human review which is useful in addressing all kinds of risks. Ultimately we build these tools with humans in mind. Supporting us is the end goal. Auditing and introducing systematic tools for review can help mitigate this type of issue. One example of a tool with addressing bias is the IBM Fairness 360 tool which can flag bias in programs.

One other risk is that of an over reliance on keywords and matching in the selection process. Approaches like TF-IDF and Jaccard Similarity heavily depend on matching exact words. Language is conplicated and these systems struggle with matching semantics and similarities. There is also a risk of misinterpreting the context of words and tokens. The classic case of this is these systems struggling with recognizing sarcasm. These systems also face difficulties with understanding nuances like soft skills or transferable experiences. These are very human contexts that require lots of context to understand that many language models lack.We can mitigate the risk of keyword reliance by expanding pre-trained embeddings to understand synonyms and related concepts such as software engineer and developer.



Another risk is that of false positives and negatives in matching. These systems are built with selection or non-selection as the end goal in mind. As such, some matching algorithms may flag irrelevant resumes as a match (false positives) or overlook qualified candidates (false negatives.The best approach to mitigating this is tuning our threshold similarity scores to reduce irrelevant matches.

