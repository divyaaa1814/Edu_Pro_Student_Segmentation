
# 🎓 EduPro: Student Segmentation & Personalized Course Recommendation System

AI-powered learner segmentation (K=4, 84.2% intra-cluster similarity) and hybrid course recommender (81.6% Precision@5) built with Python, Scikit-Learn, and Streamlit.

* **Live Streamlit Dashboard:** [http://localhost:8501](#)
* **Research Report (Google Doc):** [https://drive.google.com/drive/folders/1ylHH5blMTr3EdYmidVp2EzzKe-R2P58R]

---

## 📌 Overview

Online learners have very different learning styles, goals, and paces. Giving every student the exact same list of courses leads to low engagement, poor completion rates, and high drop-off. 

**EduPro** is an AI-powered learner segmentation and personalized course recommendation system built for **Unified Mentor** in alignment with **Toronto Government Parks, Forestry & Recreation**. Instead of treating all learners the same, EduPro analyzes student behavior, groups similar learners together using unsupervised machine learning, and recommends the most relevant courses for each individual's learning journey[cite: 1].

---

## 🎯 Objectives

* **Understand Learner Behavior:** Group 3,000 online learners into clear, meaningful segments based on their enrollments, spending, and course preferences[cite: 1].
* **Solve One-Size-Fits-All Recommendations:** Replace generic course lists with personalized, cluster-aware course suggestions.
* **Boost Engagement & Completion:** Help learners discover courses that match their current skill level and interests, driving a projected **+27.8% engagement lift**[cite: 1].
* **Provide Live Stakeholder Analytics:** Deliver an interactive web dashboard and formal documentation for government and educational decision-makers.

---

## 📊 Dataset Information

The system uses the **`EduPro Online Platform.xlsx`** dataset containing **3,000 learners**, **60 courses**, and **15,000+ recommendation mappings** across three relational sheets[cite: 1, 3]:
* **Users Sheet:** `UserID`, `Age`, `Gender`
* **Courses Sheet:** `CourseID`, `CourseCategory`, `CourseType`, `CourseLevel`, `CourseRating` (Average catalog rating: **3.13**)[cite: 1]
* **Transactions Sheet:** `UserID`, `CourseID`, `TransactionDate`, `Amount`

---

## 🛠️ Technologies Used

* **Programming Language:** Python 3.10+
* **Data Manipulation & Analysis:** Pandas, NumPy, OpenPyXL
* **Machine Learning & Clustering:** Scikit-Learn (K-Means, Agglomerative Hierarchical Clustering, PCA, StandardScaler, Cosine Similarity)
* **Data Visualization:** Plotly Express, Plotly Graph Objects
* **Web Application Framework:** Streamlit

---

## 🔍 Exploratory Data Analysis (EDA)

During EDA (`src/Notebooks/EduPro_Analysis.ipynb`)[cite: 3], we analyzed enrollment trends, course ratings, spending patterns, and difficulty distributions across all 3,000 users[cite: 1]:
* Identified wide variations between learners who only take one or two mandatory courses versus learners who explore multiple subjects.
* Discovered that course ratings average **3.13** across the catalog, making rating-weighted filtering essential so low-rated courses are not over-recommended[cite: 1].
* Observed distinct behavioral patterns across Beginner, Intermediate, and Advanced course levels.

---

## ⚙️ Feature Engineering

Raw transaction logs were aggregated at the `UserID` level to build a rich behavioral profile for each student:
* **Engagement Features:** Total courses enrolled, average courses per category, and enrollment frequency.
* **Preference Features:** Preferred course category, preferred course level, and average course rating enrolled.
* **Behavioral Features:** 
  * *Average Spending per Learner:* Total transaction amount divided by total enrollments.
  * *Diversity Score:* Number of unique course categories a learner has explored.
  * *Learning Depth Index:* Ratio of advanced-level courses taken compared to beginner-level courses.

---

## 👥 Learner Segmentation

Using unsupervised clustering, learners were divided into **4 distinct segments** (`K = 4`)[cite: 1]:
1. **Highly Engaged Explorers:** Active learners who explore many different categories and complete courses consistently[cite: 1].
2. **Advanced-Focused Learners:** Experienced students with a high Learning Depth Index who prefer challenging, specialized courses[cite: 1].
3. **Intermediate Selective Learners:** Goal-oriented learners who focus tightly on specific topics with high efficiency[cite: 1].
4. **Beginner AI Learners:** Newer learners exploring introductory tech/AI courses who benefit from guided, step-by-step learning paths[cite: 1].

---

## 🧠 Clustering Technique Used

* **K-Means Clustering:** Used as the primary partitioning algorithm (`kmeans_model.pkl`) to group standardized learner profiles into `K = 4` distinct clusters[cite: 1, 3].
* **Hierarchical Clustering (Ward's Linkage):** Used alongside K-Means to cross-validate cluster stability and hierarchical groupings[cite: 1].
* **Principal Component Analysis (PCA):** Reduced multi-dimensional behavioral features into 2 principal components (**PC1: ~34.2% variance**, **PC2: ~22.6% variance**, **Total Cumulative Variance: 56.8%**) for clear 2D cluster visualization[cite: 1].

---

## 📈 Evaluation Methods

To validate both the clustering quality and the recommendation engine, four core metrics were tracked[cite: 1]:
* **Elbow Method (Inertia):** Plotted within-cluster sum of squares across `K = 2` to `8` to identify the optimal bend at `K = 4`[cite: 1].
* **Silhouette Score:** Measured how well-separated and compact the clusters are across `K = 2` to `8`[cite: 1].
* **Intra-Cluster Similarity:** Measured behavioral consistency within each learner group (**84.2%**)[cite: 1].
* **Recommendation Precision@5:** Evaluated top-5 recommendation relevance (**81.6%**, a **+22.4%** improvement over baseline)[cite: 1].
* **Engagement Lift (Proxy):** Estimated projected improvement in learner course engagement (**+27.8%**)[cite: 1].

---

## 🏅 Best Silhouette Score

* **Best Silhouette Score:** **`0.1612`** at **`K = 4`**[cite: 1], confirming optimal cluster separation and operational interpretability across high-dimensional behavioral learner data.

---

## 🤖 Machine Learning Models Evaluated

* **K-Means Clustering (`K = 2` through `K = 8`)[cite: 1]**
* **Agglomerative Hierarchical Clustering (Ward's Linkage)[cite: 1]**
* **Content-Based & Cluster-Aware Hybrid Recommendation Scoring Models**

---

## 🏆 Best Performing Model

* **K-Means Clustering (`K = 4`)** paired with **Hierarchical Ward's Linkage cross-validation** achieved the best balance of mathematical compactness, **84.2% intra-cluster cohesion**, and clear segment interpretability for workforce training programs[cite: 1].

---

## 🎯 Recommendation System

The personalized recommendation engine uses a **cluster-aware hybrid logic**:
1. **Cluster Alignment:** Identifies the learner's assigned segment and prioritizes courses popular among similar peers in that cluster.
2. **Content-Based Similarity:** Matches course categories and difficulty levels to the student's historical preferences.
3. **Rating-Weighted Relevance:** Weights recommendations using course ratings so high-quality, well-reviewed courses rank higher.
4. **Explainability:** Displays clear match scores and relevance tags for every recommended course.

---

## 🖥️ Streamlit Dashboard Features

The live web application (`app/app.py`) includes four interactive modules[cite: 1]:
1. **📊 Segments & Analytics:** High-level KPI cards (3,000 learners, 4 segments, 15,000 recommendations, 3.13 avg rating), Executive Briefing expander for Toronto Parks & Recreation stakeholders, and segment comparison charts[cite: 1].
2. **👤 Learner Profile:** Interactive `UserID` selector displaying individual demographics, spending/course stats, assigned segment badge, and learning stage progression[cite: 1].
3. **🎯 Recommendations:** Personalized course cards with ratings and match scores, dynamic filters by **Course Category** and **Course Level**, and a one-click CSV download button[cite: 1].
4. **🧠 Clusters & Evaluation:** Interactive 2D PCA cluster scatter plot, Elbow Method chart, Silhouette Score curve, and full model validation metrics[cite: 1].

---

# Running the Project 

## Install Dependencies

'''bash
pip install -r requirements.txt
'''

## Run Streamlit Dashboard 

'''bash 
python -m streamlit run app/app.py
'''


---


## Research and Academic Notes

* **Methodology Shift:**  Traditional ed-tech analytics often focus on predicting overall course demand. EduPro shifts the focus to student-centric intelligence—understanding individual learner archetypes first and adapting course discovery around their journey.

* **Cold-Start Mitigation:**  By combining cluster-level popularity with content-based metadata, new learners with minimal transaction history still receive accurate, high-rated starter recommendations.

---


## Future Improvements 

* **Real-Time Re-Clustering:** Automate quarterly pipeline updates so learners dynamically move between segments as they complete advanced courses.

* **Deep Learning Embeddings:**  Incorporate sequential neural models (such as Transformers or collaborative matrix factorization) to capture the exact order in which courses are taken.

* **LMS Integration:** Build a REST API endpoint to push CSV/JSON recommendations directly into municipal Learning Management Systems (LMS).


---

# Author 

Divyaa Parikh\
Student of B.Sc. Data Science\
Department of Technology, Savitribai Phule University 

---


# Acknowledgement

Special thanks to Unified Mentor and the Toronto Government Parks, Forestry & Recreation initiative for providing the project framework, problem scope, and real-world workforce upskilling context.


---

## 📂 Project Structure
```text
Edu_Pro_Student_Segmention/
│
├── app/
│   └── app.py                         # Main Streamlit dashboard application
│
├── src/
│   ├── Data/
│   │   └── EduPro Online Platform.xlsx # Raw dataset (Users, Courses, Transactions)
│   │
│   ├── models/
│   │   ├── encoder.pkl                # Fitted categorical encoder
│   │   ├── kmeans_model.pkl           # Trained K-Means clustering model (K=4)
│   │   └── scaler.pkl                 # Fitted numerical feature scaler
│   │
│   ├── Notebooks/
│   │   └── EduPro_Analysis.ipynb      # EDA, feature engineering, clustering & evaluation notebook
│   │
│   └── outputs/
│       ├── cluster_summary.csv        # Statistical summary metrics per segment
│       ├── learner_profiles.csv       # Learner profiles with engineered features & cluster IDs
│       └── recommendations.csv        # Pre-computed personalized course recommendations
│
├── edu_pro.gitignore                  # Git ignore rules
├── Executive_Summary.md               # Stakeholder summary brief
├── README.md                          # Project documentation
└── requirements.txt                   # Python libraries required to run the project
'''

--- 



