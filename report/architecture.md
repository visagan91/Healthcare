flowchart TB
  %% =====================
  %% DATA SOURCES
  %% =====================
  subgraph S[Data Sources]
    A1[Master Tabular Dataset\n(data/master_healthcare_dataset.csv)]
    A2[X-ray Dataset\n(images/xray_images + xray_labels.csv)]
  end

  %% =====================
  %% INGESTION + QA
  %% =====================
  subgraph I[Ingestion & Data Quality]
    B1[Load & Validate Schema\n- dtypes\n- ranges\n- missingness]
    B2[Split Train/Val/Test\n(using data_split)]
    B3[Preprocessing Pipelines\n- impute\n- scale\n- one-hot]
  end

  %% =====================
  %% FEATURE LAYERS
  %% =====================
  subgraph F[Feature Layers]
    C1[Tabular Features\nVitals/Labs/Demo/Lifestyle]
    C2[Text Features\nNotes + Discharge + Feedback]
    C3[Time-series Features\nvitals_ts_json + labs_ts_json]
    C4[Imaging Features (Runtime)\nCNN embeddings from X-rays]
    C5[Association Basket\ntransaction_basket]
  end

  %% =====================
  %% MODEL LAYER
  %% =====================
  subgraph M[Model Layer]
    D1[Task 1: Regression\nPredict LOS (los_days)]
    D2[Task 2: Classification\nRisk category (risk_category)]
    D3[Task 3: Clustering\nPatient subgroups (cluster_id)]
    D4[Task 4: Assoc. Rules\nApriori/FP-Growth rules]
    D5[Task 5: Deep Learning\nNN (tabular), RNN/LSTM (time-series), CNN (images)]
    D6[Task 6: Pretrained NLP\nClinicalBERT/BioBERT embeddings]
    D7[Task 9: Sentiment Model\nFeedback sentiment_label]
  end

  %% =====================
  %% CHATBOT + TRANSLATION
  %% =====================
  subgraph C[Patient Engagement Layer]
    E1[RAG Store\nText Embeddings Index]
    E2[Healthcare Chatbot\npatient_question -> response]
    E3[Reference Answer Comparison\nchatbot_reference_answer]
  end

  %% =====================
  %% OUTPUTS
  %% =====================
  subgraph O[Outputs & Decision Support]
    F1[Predictions Dashboard\nLOS, Risk, Readmit, Mortality]
    F2[Patient Segments\nCluster profiles]
    F3[Care Pathway Patterns\nAssociation rules]
    F4[Patient Experience Insights\nSentiment trends]
    F5[Chatbot Responses\nPatient Q&A]
  end

  %% =====================
  %% FLOW
  %% =====================
  A1 --> B1 --> B2 --> B3
  A2 --> C4

  B3 --> C1
  A1 --> C2
  A1 --> C3
  A1 --> C5

  C1 --> D1
  C1 --> D2
  C1 --> D3
  C5 --> D4
  C1 --> D5
  C3 --> D5
  C4 --> D5
  C2 --> D6
  C2 --> D7

  D6 --> E1
  E1 --> E2
  A1 --> E2
  E2 --> E3

  D1 --> F1
  D2 --> F1
  D3 --> F2
  D4 --> F3
  D7 --> F4
  E2 --> F5
