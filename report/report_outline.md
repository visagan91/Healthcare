# Title
End-to-End AI/ML Clinical Decision Support System Using Multi-Modal Patient Data

## 1. Introduction
- Motivation: decision support + patient engagement + ops efficiency
- Problem statement (9 tasks)
- Contributions (single integrated system, multi-paradigm)

## 2. Dataset Design & Sources
- Master dataset: synthetic but clinically structured (5606 rows)
- Imaging dataset: external X-ray dataset loaded at runtime
- Key columns & why they exist (e.g., transaction_basket, chatbot_reference_answer)
- Missingness strategy (realistic NaNs)

## 3. Data Preprocessing
- Train/val/test split
- Numeric preprocessing: median imputation + scaling
- Categorical preprocessing: mode imputation + one-hot
- Text preprocessing: cleaning + tokenization (for later tasks)
- Time-series parsing: vitals_ts_json / labs_ts_json

## 4. System Architecture
- Diagram + explanation of modules
- Data flow and separation of modalities

## 5. Task 1: Outcome Prediction (Regression)
- Target: los_days
- Models: linear regression baseline vs random forest
- Metrics: MAE, RMSE, R²
- Key findings

## 6. Task 2: Disease Risk Classification
- Target: risk_category (low/medium/high)
- Models: logistic regression vs random forest
- Metrics: accuracy, F1, ROC-AUC OvR
- Confusion matrix + interpretation

## 7. Task 3: Patient Subgroup Discovery (Clustering)
- Features used + scaling
- K selection (elbow + silhouette)
- Cluster profiles + clinical interpretation

## 8. Task 4: Association Rule Mining
- Transaction construction (transaction_basket)
- FP-Growth / Apriori
- Support/confidence/lift
- Example clinical rules

## 9. Task 5: Deep Learning Models
- NN for tabular classification
- RNN/LSTM for time-series prediction
- CNN pipeline on X-ray images (concept + small demo)
- Comparison vs classical models

## 10. Task 6: Pretrained Models (BioBERT/ClinicalBERT)
- Embedding generation from notes/discharge summaries
- Downstream tasks: classification/retrieval
- Benefits and limitations

## 11. Task 7: Healthcare Chatbot
- RAG pipeline using embeddings
- Prompting + safety constraints
- Reference answer comparison (chatbot_reference_answer)

## 12. Task 9: Sentiment Analysis on Patient Feedback
- TF-IDF + logistic regression baseline
- BERT-based classifier (optional)
- Metrics + insights

## 13. Discussion
- Clinical utility
- Operational impact (LOS/cost/readmissions)
- Patient engagement benefits

## 14. Ethics, Privacy, and Limitations
- Synthetic data note
- Bias risks
- Safety constraints (chatbot)
- Generalization limitations

## 15. Conclusion & Future Work
- Summary of results
- Next steps: real-world validation, multimodal fusion, deployment
