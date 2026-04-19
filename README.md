# Customer-Churn-Predictor

🧠 ANN Customer Churn Prediction
Goal Built a deep learning model to predict which telecom customers are likely to churn, enabling proactive retention before revenue is lost.

Tech Stack Python, TensorFlow, Keras, scikit-learn, pandas, NumPy, ANN (Sequential), Dropout Regularization, StandardScaler, Label Encoding.

Challenges & Solutions 

  • Mixed feature types → Label-encoded categoricals and StandardScaler-normalized numerics for stable gradient descent 

  • Overfitting on small dataset → Applied Dropout layers (30/30/20%) across all hidden layers to force generalization 

  • Training instability → Used EarlyStopping (patience=15) + ModelCheckpoint to lock in peak weights and prevent late-epoch regression 
  
  • Class imbalance → Stratified train/test split and evaluated with ROC-AUC + full classification report rather than raw accuracy
  
Impact 

Delivered a production-ready binary classifier with a 3-layer ANN architecture that outputs churn probabilities per customer, giving retention teams a ranked at-risk list to act on before churn occurs.
