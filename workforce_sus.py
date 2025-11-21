

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ All libraries imported successfully!")

# %% [markdown]
# ## 2. Load Claims Data

# %%
# Load the claims data
print("Loading claims data...")
claims_df = pd.read_csv('Fianl_claims_data.csv')

# Convert date columns
date_columns = ['Incurred Date', 'Paid Date', 'Contract Start Date', 
                'Contract End Date', 'Admission Date', 'Discharge Date']

for col in date_columns:
    if col in claims_df.columns:
        claims_df[col] = pd.to_datetime(claims_df[col], dayfirst=True, errors='coerce')

print(f"✓ Loaded {len(claims_df):,} claims")
print(f"✓ {claims_df['Unique Member Reference'].nunique():,} unique members")
print(f"✓ Date range: {claims_df['Incurred Date'].min()} to {claims_df['Incurred Date'].max()}")

# Display first few rows
claims_df.head()

# %% [markdown]
# ## 3. Exploratory Data Analysis

# %%
# Basic statistics
print("="*60)
print("CLAIMS DATA SUMMARY")
print("="*60)
print(f"\nTotal Claims: {len(claims_df):,}")
print(f"Unique Members: {claims_df['Unique Member Reference'].nunique():,}")
print(f"Unique Claimants: {claims_df['Claimant Unique ID'].nunique():,}")
print(f"\nDate Range: {claims_df['Incurred Date'].min().date()} to {claims_df['Incurred Date'].max().date()}")
print(f"\nTotal Claim Amount: £{claims_df['Claim Amount'].sum():,.2f}")
print(f"Average Claim Amount: £{claims_df['Claim Amount'].mean():,.2f}")
print(f"Median Claim Amount: £{claims_df['Claim Amount'].median():,.2f}")

# %%
# Condition distribution
print("\nCondition Category Distribution:")
condition_dist = claims_df['Condition Category'].value_counts()
print(condition_dist.head(10))

# Visualize
plt.figure(figsize=(12, 6))
condition_dist.head(10).plot(kind='bar')
plt.title('Top 10 Condition Categories', fontsize=14, fontweight='bold')
plt.xlabel('Condition Category')
plt.ylabel('Number of Claims')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Claim type distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Claim Type
claims_df['Claim Type'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Claim Type Distribution', fontweight='bold')
axes[0].set_xlabel('Claim Type')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Treatment Type
claims_df['Treatment Type'].value_counts().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Treatment Type Distribution', fontweight='bold')
axes[1].set_xlabel('Treatment Type')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
# Age distribution
plt.figure(figsize=(10, 6))
claims_df['Age'].hist(bins=30, edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Claimants', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.axvline(claims_df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {claims_df["Age"].mean():.1f}')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %%
# Claims by year
claims_df['Year'] = claims_df['Incurred Date'].dt.year
yearly_claims = claims_df.groupby('Year').agg({
    'Claim ID': 'count',
    'Claim Amount': 'sum'
}).rename(columns={'Claim ID': 'Claim Count', 'Claim Amount': 'Total Amount'})

print("\nClaims by Year:")
print(yearly_claims)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

yearly_claims['Claim Count'].plot(kind='bar', ax=axes[0], color='green')
axes[0].set_title('Claims Count by Year', fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of Claims')

yearly_claims['Total Amount'].plot(kind='bar', ax=axes[1], color='purple')
axes[1].set_title('Total Claim Amount by Year', fontweight='bold')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Amount (£)')
axes[1].ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.show()

claims_df = claims_df.drop('Year', axis=1)  # Clean up temporary column

# %% [markdown]
# ## 4. Feature Engineering

# %%
def engineer_features(claims_df, prediction_date=None, verbose=True):
    """
    Engineer features from claims history for each member
    """
    
    if prediction_date is None:
        prediction_date = claims_df['Incurred Date'].max()
    
    if verbose:
        print(f"\nEngineering features up to {prediction_date.strftime('%Y-%m-%d')}...")
    
    # Filter claims before prediction date
    historical_claims = claims_df[claims_df['Incurred Date'] <= prediction_date].copy()
    
    member_features = []
    
    for i, member_id in enumerate(historical_claims['Unique Member Reference'].unique()):
        if verbose and (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,} members...")
        
        member_claims = historical_claims[
            historical_claims['Unique Member Reference'] == member_id
        ].sort_values('Incurred Date')
        
        if len(member_claims) == 0:
            continue
        
        # Get latest member info
        latest_claim = member_claims.iloc[-1]
        
        # Calculate time windows
        last_claim_date = member_claims['Incurred Date'].max()
        lookback_6m = last_claim_date - timedelta(days=180)
        lookback_12m = last_claim_date - timedelta(days=365)
        lookback_24m = last_claim_date - timedelta(days=730)
        
        # Claims in different time windows
        claims_6m = member_claims[member_claims['Incurred Date'] >= lookback_6m]
        claims_12m = member_claims[member_claims['Incurred Date'] >= lookback_12m]
        claims_24m = member_claims[member_claims['Incurred Date'] >= lookback_24m]
        
        # === DEMOGRAPHIC FEATURES ===
        age = latest_claim.get('Age', 40)
        gender_encoded = 1 if latest_claim.get('Gender') == 'Male' else 0
        
        # === TEMPORAL FEATURES ===
        claims_count_6m = len(claims_6m)
        claims_count_12m = len(claims_12m)
        claims_count_24m = len(claims_24m)
        
        # Claim frequency trend
        if claims_count_6m > claims_count_12m * 0.6:
            claim_trend = 1  # Increasing
        elif claims_count_6m < claims_count_12m * 0.3:
            claim_trend = -1  # Decreasing
        else:
            claim_trend = 0  # Stable
        
        # Days since last claim
        days_since_last_claim = (prediction_date - last_claim_date).days
        
        # === TREATMENT INTENSITY FEATURES ===
        total_claim_amount_12m = claims_12m['Claim Amount'].sum()
        avg_claim_amount_12m = claims_12m['Claim Amount'].mean()
        max_claim_amount_12m = claims_12m['Claim Amount'].max()
        
        # Hospitalization rate
        inpatient_claims = claims_12m[claims_12m['Claim Type'] == 'Inpatient']
        hospitalization_rate = len(inpatient_claims) / len(claims_12m) if len(claims_12m) > 0 else 0
        
        # Average length of stay
        avg_los = claims_12m['Calculate Length of Service'].mean()
        max_los = claims_12m['Calculate Length of Service'].max()
        
        # === DIAGNOSTIC & TREATMENT PATTERN FEATURES ===
        diagnostic_claims = claims_12m[claims_12m['Treatment Type'] == 'Diagnostic']
        surgical_claims = claims_12m[claims_12m['Treatment Type'] == 'Surgical']
        therapeutic_claims = claims_12m[claims_12m['Treatment Type'] == 'Therapeutic']
        
        diagnostic_ratio = len(diagnostic_claims) / len(claims_12m) if len(claims_12m) > 0 else 0
        surgical_ratio = len(surgical_claims) / len(claims_12m) if len(claims_12m) > 0 else 0
        therapeutic_ratio = len(therapeutic_claims) / len(claims_12m) if len(claims_12m) > 0 else 0
        
        # Specialist visits
        hospital_visits = claims_12m[claims_12m['Provider Type'] == 'Hospital']
        specialist_ratio = len(hospital_visits) / len(claims_12m) if len(claims_12m) > 0 else 0
        
        # Diagnostic center visits (imaging/tests)
        diagnostic_center_visits = len(claims_12m[claims_12m['Provider Type'] == 'Diagnostic Center'])
        
        # === ANCILLARY SERVICE FEATURES ===
        ancillary_services = claims_12m['Ancillary Service Type'].dropna().tolist()
        has_imaging = any('Imaging' in str(s) for s in ancillary_services)
        has_pathology = any('Pathology' in str(s) for s in ancillary_services)
        has_therapy = any('Therapy' in str(s) for s in ancillary_services)
        has_chemotherapy = any('Chemotherapy' in str(s) for s in ancillary_services)
        has_radiotherapy = any('Radiotherapy' in str(s) for s in ancillary_services)
        has_physiotherapy = any('Physiotherapy' in str(s) for s in ancillary_services)
        
        # === CONDITION HISTORY FEATURES ===
        all_conditions = member_claims['Condition Category'].unique().tolist()
        
        # Pre-existing condition indicators
        has_musculoskeletal = 'Musculoskeletal' in all_conditions
        has_digestive = 'Digestive' in all_conditions
        has_respiratory = 'Respiratory' in all_conditions
        has_cardiovascular = 'Cardiovascular' in all_conditions
        has_mental_health = 'Mental Health' in all_conditions
        has_oncology = 'Oncology' in all_conditions
        has_endocrine = 'Endocrine' in all_conditions
        has_neurology = 'Neurology' in all_conditions
        has_orthopaedics = 'Orthopaedics' in all_conditions
        has_rheumatology = 'Rheumatology' in all_conditions
        
        # Count unique condition categories
        unique_conditions_count = len(all_conditions)
        
        # === RISK INDICATORS ===
        # High claim amount indicator
        high_claim_indicator = 1 if avg_claim_amount_12m > 5000 else 0
        
        # Frequent claimer
        frequent_claimer = 1 if claims_count_12m > 8 else 0
        
        # Multiple hospitalizations
        multiple_hospitalizations = 1 if len(inpatient_claims) > 2 else 0
        
        # Escalating care (increase in claim amounts)
        if len(claims_12m) >= 2:
            first_half = claims_12m.iloc[:len(claims_12m)//2]['Claim Amount'].mean()
            second_half = claims_12m.iloc[len(claims_12m)//2:]['Claim Amount'].mean()
            escalating_costs = 1 if second_half > first_half * 1.3 else 0
        else:
            escalating_costs = 0
        
        # === TARGET VARIABLE ===
        # Check if member develops high-risk condition in future
        future_claims = claims_df[
            (claims_df['Unique Member Reference'] == member_id) &
            (claims_df['Incurred Date'] > prediction_date) &
            (claims_df['Incurred Date'] <= prediction_date + timedelta(days=365))
        ]
        
        high_risk_conditions = ['Oncology', 'Cardiovascular', 'Mental Health']
        
        # Target: Will develop high-risk condition in next 12 months?
        develops_high_risk = any(
            cond in high_risk_conditions 
            for cond in future_claims['Condition Category'].unique()
        )
        
        # Specific condition targets
        develops_oncology = 'Oncology' in future_claims['Condition Category'].values
        develops_cardiovascular = 'Cardiovascular' in future_claims['Condition Category'].values
        develops_mental_health = 'Mental Health' in future_claims['Condition Category'].values
        
        # Compile features
        features = {
            'member_id': member_id,
            'age': age,
            'gender': gender_encoded,
            'claims_count_6m': claims_count_6m,
            'claims_count_12m': claims_count_12m,
            'claims_count_24m': claims_count_24m,
            'claim_trend': claim_trend,
            'days_since_last_claim': days_since_last_claim,
            'total_claim_amount_12m': total_claim_amount_12m,
            'avg_claim_amount_12m': avg_claim_amount_12m,
            'max_claim_amount_12m': max_claim_amount_12m,
            'hospitalization_rate': hospitalization_rate,
            'avg_los': avg_los,
            'max_los': max_los,
            'diagnostic_ratio': diagnostic_ratio,
            'surgical_ratio': surgical_ratio,
            'therapeutic_ratio': therapeutic_ratio,
            'specialist_ratio': specialist_ratio,
            'diagnostic_center_visits': diagnostic_center_visits,
            'has_imaging': int(has_imaging),
            'has_pathology': int(has_pathology),
            'has_therapy': int(has_therapy),
            'has_chemotherapy': int(has_chemotherapy),
            'has_radiotherapy': int(has_radiotherapy),
            'has_physiotherapy': int(has_physiotherapy),
            'has_musculoskeletal': int(has_musculoskeletal),
            'has_digestive': int(has_digestive),
            'has_respiratory': int(has_respiratory),
            'has_cardiovascular': int(has_cardiovascular),
            'has_mental_health': int(has_mental_health),
            'has_oncology': int(has_oncology),
            'has_endocrine': int(has_endocrine),
            'has_neurology': int(has_neurology),
            'has_orthopaedics': int(has_orthopaedics),
            'has_rheumatology': int(has_rheumatology),
            'unique_conditions_count': unique_conditions_count,
            'high_claim_indicator': high_claim_indicator,
            'frequent_claimer': frequent_claimer,
            'multiple_hospitalizations': multiple_hospitalizations,
            'escalating_costs': escalating_costs,
            # Targets
            'develops_high_risk': int(develops_high_risk),
            'develops_oncology': int(develops_oncology),
            'develops_cardiovascular': int(develops_cardiovascular),
            'develops_mental_health': int(develops_mental_health)
        }
        
        member_features.append(features)
    
    features_df = pd.DataFrame(member_features)
    
    if verbose:
        print(f"✓ Engineered features for {len(features_df):,} members")
        print(f"✓ High-risk condition rate: {features_df['develops_high_risk'].mean()*100:.2f}%")
    
    return features_df

# %%
# Define training cutoff date (we'll train on data before this date)
training_cutoff = pd.to_datetime('2023-01-30')

print("="*60)
print("FEATURE ENGINEERING - TRAINING SET")
print("="*60)
print(f"Training data cutoff: {training_cutoff.strftime('%Y-%m-%d')}")
print(f"We'll predict conditions developing in the 12 months after this date")

# Engineer features
training_features = engineer_features(claims_df, prediction_date=training_cutoff)

# Display sample features
print("\nSample of engineered features:")
training_features.head()

# %% [markdown]
# ## 5. Prepare Training Data

# %%
# Select feature columns (exclude ID and target columns)
exclude_cols = ['member_id', 'develops_high_risk', 'develops_oncology', 
               'develops_cardiovascular', 'develops_mental_health']

feature_columns = [col for col in training_features.columns if col not in exclude_cols]

print(f"Total features: {len(feature_columns)}")
print("\nFeature list:")
for i, col in enumerate(feature_columns, 1):
    print(f"{i}. {col}")

# %%
# Prepare X and y
X = training_features[feature_columns].fillna(0)
y = training_features['develops_high_risk']

print("\n" + "="*60)
print("TARGET VARIABLE DISTRIBUTION")
print("="*60)
print(f"Total members: {len(y):,}")
print(f"Develops high-risk condition: {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"Does not develop: {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.2f}%)")

# Visualize class distribution
fig, ax = plt.subplots(figsize=(8, 5))
y.value_counts().plot(kind='bar', color=['green', 'red'], ax=ax)
ax.set_title('Target Variable Distribution', fontweight='bold')
ax.set_xlabel('Develops High-Risk Condition')
ax.set_xticklabels(['No', 'Yes'], rotation=0)
ax.set_ylabel('Count')
for i, v in enumerate(y.value_counts().values):
    ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*60)
print("TRAIN/VALIDATION SPLIT")
print("="*60)
print(f"Training set: {len(X_train):,} members")
print(f"  - High risk: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"  - Low risk: {(~y_train.astype(bool)).sum():,}")

print(f"\nValidation set: {len(X_val):,} members")
print(f"  - High risk: {y_val.sum():,} ({y_val.mean()*100:.2f}%)")
print(f"  - Low risk: {(~y_val.astype(bool)).sum():,}")

# %% [markdown]
# ## 6. Train Naive Bayes Model

# %%
print("="*60)
print("TRAINING NAIVE BAYES MODEL")
print("="*60)

# Initialize and train model
model = GaussianNB()
model.fit(X_train, y_train)

print("✓ Model trained successfully!")
print(f"✓ Model type: {type(model).__name__}")
print(f"✓ Number of features: {X_train.shape[1]}")
print(f"✓ Number of classes: {len(model.classes_)}")

# %% [markdown]
# ## 7. Model Evaluation

# %%
# Make predictions on validation set
y_pred_val = model.predict(X_val)
y_pred_proba_val = model.predict_proba(X_val)

print("="*60)
print("MODEL PERFORMANCE ON VALIDATION SET")
print("="*60)

# Accuracy
accuracy = accuracy_score(y_val, y_pred_val)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val, 
                          target_names=['No High Risk', 'High Risk'],
                          digits=4))

# %%
# Confusion Matrix
cm = confusion_matrix(y_val, y_pred_val)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No High Risk', 'High Risk'],
            yticklabels=['No High Risk', 'High Risk'])
plt.title('Confusion Matrix - Validation Set', fontweight='bold', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

print("\nConfusion Matrix Interpretation:")
print(f"True Negatives (TN): {cm[0,0]:,} - Correctly predicted no high risk")
print(f"False Positives (FP): {cm[0,1]:,} - Incorrectly predicted high risk")
print(f"False Negatives (FN): {cm[1,0]:,} - Missed high risk cases")
print(f"True Positives (TP): {cm[1,1]:,} - Correctly predicted high risk")

# %%
# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_val[:, 1])
roc_auc = roc_auc_score(y_val, y_pred_proba_val[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Disease Prediction Model', fontweight='bold', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ ROC-AUC Score: {roc_auc:.4f}")
print("✓ ROC curve saved as 'roc_curve.png'")

# %%
# Precision-Recall Curve
precision, recall, pr_thresholds = precision_recall_curve(y_val, y_pred_proba_val[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontweight='bold', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Feature Importance (based on difference in class means)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': np.abs(model.theta_[1] - model.theta_[0])
}).sort_values('importance', ascending=False)

print("="*60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*60)
print(feature_importance.head(15).to_string(index=False))

# Visualize
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance Score')
plt.title('Top 15 Most Important Features', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Train Individual Disease Models
# 
# Now we'll train separate models for each disease type to get individual likelihoods

# %%
print("="*60)
print("TRAINING DISEASE-SPECIFIC MODELS")
print("="*60)

# Prepare data for each disease
disease_models = {}
disease_targets = {
    'Oncology': 'develops_oncology',
    'Cardiovascular': 'develops_cardiovascular',
    'Mental Health': 'develops_mental_health'
}

# Train a model for each disease
for disease_name, target_col in disease_targets.items():
    print(f"\nTraining model for {disease_name}...")
    
    # Prepare target
    y_disease = training_features[target_col]
    
    # Split data
    X_train_disease, X_val_disease, y_train_disease, y_val_disease = train_test_split(
        X, y_disease, test_size=0.2, random_state=42, stratify=y_disease
    )
    
    # Train model
    model_disease = GaussianNB()
    model_disease.fit(X_train_disease, y_train_disease)
    
    # Evaluate
    y_pred_disease = model_disease.predict(X_val_disease)
    y_pred_proba_disease = model_disease.predict_proba(X_val_disease)
    
    accuracy = accuracy_score(y_val_disease, y_pred_disease)
    
    try:
        auc = roc_auc_score(y_val_disease, y_pred_proba_disease[:, 1])
        print(f"  ✓ Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    except:
        print(f"  ✓ Accuracy: {accuracy:.4f}")
    
    print(f"  ✓ Positive cases: {y_disease.sum()} ({y_disease.mean()*100:.2f}%)")
    
    # Store model
    disease_models[disease_name] = model_disease

print("\n✓ All disease-specific models trained!")

# %%
# Evaluate each disease model in detail
print("\n" + "="*60)
print("DISEASE-SPECIFIC MODEL PERFORMANCE")
print("="*60)

for disease_name, target_col in disease_targets.items():
    print(f"\n{disease_name.upper()} MODEL")
    print("-" * 40)
    
    model_disease = disease_models[disease_name]
    y_disease = training_features[target_col]
    
    X_train_disease, X_val_disease, y_train_disease, y_val_disease = train_test_split(
        X, y_disease, test_size=0.2, random_state=42, stratify=y_disease
    )
    
    y_pred_disease = model_disease.predict(X_val_disease)
    y_pred_proba_disease = model_disease.predict_proba(X_val_disease)
    
    print(classification_report(y_val_disease, y_pred_disease, 
                               target_names=['No', 'Yes'], digits=3))

# %% [markdown]
# ## 9. Make Predictions for All Diseases

# %%
# Engineer features for prediction (using more recent data)
prediction_cutoff = pd.to_datetime('2023-12-31')

print("="*60)
print("PREDICTIONS ON NEW MEMBERS - ALL DISEASES")
print("="*60)
print(f"Prediction data cutoff: {prediction_cutoff.strftime('%Y-%m-%d')}")

prediction_features = engineer_features(claims_df, prediction_date=prediction_cutoff)

# %%
# Prepare prediction data
X_pred = prediction_features[feature_columns].fillna(0)

# Make predictions for general high risk
predictions = model.predict(X_pred)
probabilities = model.predict_proba(X_pred)

# Create comprehensive risk report
risk_report = prediction_features[['member_id']].copy()

# Overall high risk prediction
risk_report['overall_high_risk'] = predictions
risk_report['overall_risk_probability'] = probabilities[:, 1]

# Individual disease predictions
for disease_name, disease_model in disease_models.items():
    disease_pred = disease_model.predict(X_pred)
    disease_proba = disease_model.predict_proba(X_pred)
    
    risk_report[f'{disease_name.lower().replace(" ", "_")}_prediction'] = disease_pred
    risk_report[f'{disease_name.lower().replace(" ", "_")}_likelihood'] = disease_proba[:, 1]

# Risk categorization for overall risk
risk_report['risk_category'] = pd.cut(
    risk_report['overall_risk_probability'],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Add key member information
risk_report['age'] = prediction_features['age'].values
risk_report['gender'] = prediction_features['gender'].map({1: 'Male', 0: 'Female'})
risk_report['claims_12m'] = prediction_features['claims_count_12m'].values
risk_report['avg_claim_amount'] = prediction_features['avg_claim_amount_12m'].values
risk_report['hospitalization_rate'] = prediction_features['hospitalization_rate'].values

# Add condition history flags
risk_report['has_cardiovascular_history'] = prediction_features['has_cardiovascular'].values
risk_report['has_oncology_history'] = prediction_features['has_oncology'].values
risk_report['has_mental_health_history'] = prediction_features['has_mental_health'].values

print(f"✓ Generated predictions for {len(risk_report):,} members")

# Display sample predictions with all disease likelihoods
print("\nSample Predictions (First 10 Members):")
display_cols = ['member_id', 'age', 'gender', 'overall_risk_probability', 
                'oncology_likelihood', 'cardiovascular_likelihood', 
                'mental_health_likelihood', 'risk_category']
print(risk_report[display_cols].head(10).to_string(index=False))

# %%
# Show detailed view for a specific high-risk member
print("\n" + "="*60)
print("DETAILED EXAMPLE: HIGH-RISK MEMBER ANALYSIS")
print("="*60)

# Find a member with high overall risk
high_risk_member = risk_report.nlargest(1, 'overall_risk_probability').iloc[0]

print(f"\nMember ID: {high_risk_member['member_id']}")
print(f"Age: {high_risk_member['age']}")
print(f"Gender: {high_risk_member['gender']}")
print(f"Claims (12m): {high_risk_member['claims_12m']}")
print(f"Avg Claim Amount: £{high_risk_member['avg_claim_amount']:,.2f}")
print(f"Hospitalization Rate: {high_risk_member['hospitalization_rate']:.2%}")

print(f"\n{'Disease Type':<20} {'Likelihood':<15} {'Risk Level'}")
print("-" * 50)

# Overall risk
overall_prob = high_risk_member['overall_risk_probability']
print(f"{'Overall High Risk':<20} {overall_prob:<15.2%} {high_risk_member['risk_category']}")

# Individual diseases
print(f"{'Oncology':<20} {high_risk_member['oncology_likelihood']:<15.2%} {'High' if high_risk_member['oncology_likelihood'] > 0.5 else 'Low'}")
print(f"{'Cardiovascular':<20} {high_risk_member['cardiovascular_likelihood']:<15.2%} {'High' if high_risk_member['cardiovascular_likelihood'] > 0.5 else 'Low'}")
print(f"{'Mental Health':<20} {high_risk_member['mental_health_likelihood']:<15.2%} {'High' if high_risk_member['mental_health_likelihood'] > 0.5 else 'Low'}")

print(f"\nCondition History:")
print(f"  - Cardiovascular: {'Yes' if high_risk_member['has_cardiovascular_history'] else 'No'}")
print(f"  - Oncology: {'Yes' if high_risk_member['has_oncology_history'] else 'No'}")
print(f"  - Mental Health: {'Yes' if high_risk_member['has_mental_health_history'] else 'No'}")

# %%
# Visualize disease likelihoods for top 20 high-risk members
top_20_risk = risk_report.nlargest(20, 'overall_risk_probability')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Oncology
axes[0].barh(range(len(top_20_risk)), top_20_risk['oncology_likelihood'], color='purple')
axes[0].set_yticks(range(len(top_20_risk)))
axes[0].set_yticklabels([f"Member {i+1}" for i in range(len(top_20_risk))])
axes[0].set_xlabel('Likelihood')
axes[0].set_title('Oncology Risk - Top 20 Members', fontweight='bold')
axes[0].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
axes[0].legend()

# Cardiovascular
axes[1].barh(range(len(top_20_risk)), top_20_risk['cardiovascular_likelihood'], color='red')
axes[1].set_yticks(range(len(top_20_risk)))
axes[1].set_yticklabels([f"Member {i+1}" for i in range(len(top_20_risk))])
axes[1].set_xlabel('Likelihood')
axes[1].set_title('Cardiovascular Risk - Top 20 Members', fontweight='bold')
axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
axes[1].legend()

# Mental Health
axes[2].barh(range(len(top_20_risk)), top_20_risk['mental_health_likelihood'], color='blue')
axes[2].set_yticks(range(len(top_20_risk)))
axes[2].set_yticklabels([f"Member {i+1}" for i in range(len(top_20_risk))])
axes[2].set_xlabel('Likelihood')
axes[2].set_title('Mental Health Risk - Top 20 Members', fontweight='bold')
axes[2].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
axes[2].legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Disease Likelihood Distribution Analysis

# %%
# Analyze distribution of disease likelihoods
print("="*60)
print("DISEASE LIKELIHOOD DISTRIBUTIONS")
print("="*60)

for disease in ['oncology', 'cardiovascular', 'mental_health']:
    col_name = f'{disease}_likelihood'
    print(f"\n{disease.replace('_', ' ').title()} Likelihood:")
    print(f"  Mean: {risk_report[col_name].mean():.4f}")
    print(f"  Median: {risk_report[col_name].median():.4f}")
    print(f"  Std: {risk_report[col_name].std():.4f}")
    print(f"  Min: {risk_report[col_name].min():.4f}")
    print(f"  Max: {risk_report[col_name].max():.4f}")
    print(f"  Members with >50% likelihood: {(risk_report[col_name] > 0.5).sum():,} ({(risk_report[col_name] > 0.5).mean()*100:.2f}%)")

# %%
# Visualize disease likelihood distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Oncology
axes[0, 0].hist(risk_report['oncology_likelihood'], bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(risk_report['oncology_likelihood'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {risk_report["oncology_likelihood"].mean():.3f}')
axes[0, 0].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='50% Threshold')
axes[0, 0].set_title('Oncology Likelihood Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Likelihood')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Cardiovascular
axes[0, 1].hist(risk_report['cardiovascular_likelihood'], bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(risk_report['cardiovascular_likelihood'].mean(), color='blue', linestyle='--',
                   label=f'Mean: {risk_report["cardiovascular_likelihood"].mean():.3f}')
axes[0, 1].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='50% Threshold')
axes[0, 1].set_title('Cardiovascular Likelihood Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Likelihood')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Mental Health
axes[1, 0].hist(risk_report['mental_health_likelihood'], bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(risk_report['mental_health_likelihood'].mean(), color='red', linestyle='--',
                   label=f'Mean: {risk_report["mental_health_likelihood"].mean():.3f}')
axes[1, 0].axvline(0.5, color='green', linestyle='--', alpha=0.5, label='50% Threshold')
axes[1, 0].set_title('Mental Health Likelihood Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Likelihood')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Overall comparison
disease_data = [
    risk_report['oncology_likelihood'],
    risk_report['cardiovascular_likelihood'],
    risk_report['mental_health_likelihood']
]
axes[1, 1].boxplot(disease_data, labels=['Oncology', 'Cardiovascular', 'Mental Health'])
axes[1, 1].set_title('Disease Likelihood Comparison', fontweight='bold')
axes[1, 1].set_ylabel('Likelihood')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %%
# High-risk members for each disease
print("\n" + "="*60)
print("HIGH-RISK MEMBERS BY DISEASE TYPE")
print("="*60)

threshold = 0.5

for disease in ['oncology', 'cardiovascular', 'mental_health']:
    col_name = f'{disease}_likelihood'
    high_risk_count = (risk_report[col_name] > threshold).sum()
    high_risk_pct = (risk_report[col_name] > threshold).mean() * 100
    
    print(f"\n{disease.replace('_', ' ').title()}:")
    print(f"  High Risk Members (>{threshold*100}%): {high_risk_count:,} ({high_risk_pct:.2f}%)")
    
    if high_risk_count > 0:
        high_risk_members = risk_report[risk_report[col_name] > threshold]
        print(f"  Average Age: {high_risk_members['age'].mean():.1f}")
        print(f"  Average Claims (12m): {high_risk_members['claims_12m'].mean():.1f}")
        print(f"  Average Likelihood: {high_risk_members[col_name].mean():.2%}")

# %%
# Members at high risk for multiple diseases
print("\n" + "="*60)
print("MEMBERS AT HIGH RISK FOR MULTIPLE DISEASES")
print("="*60)

risk_report['high_risk_oncology'] = (risk_report['oncology_likelihood'] > 0.5).astype(int)
risk_report['high_risk_cardiovascular'] = (risk_report['cardiovascular_likelihood'] > 0.5).astype(int)
risk_report['high_risk_mental_health'] = (risk_report['mental_health_likelihood'] > 0.5).astype(int)

risk_report['total_high_risk_diseases'] = (
    risk_report['high_risk_oncology'] + 
    risk_report['high_risk_cardiovascular'] + 
    risk_report['high_risk_mental_health']
)

print("\nMembers by Number of High-Risk Diseases:")
multi_disease = risk_report['total_high_risk_diseases'].value_counts().sort_index()
for num_diseases, count in multi_disease.items():
    print(f"  {num_diseases} disease(s): {count:,} members ({count/len(risk_report)*100:.2f}%)")

# Members at risk for all 3 diseases
triple_risk = risk_report[risk_report['total_high_risk_diseases'] == 3]
if len(triple_risk) > 0:
    print(f"\n⚠ CRITICAL: {len(triple_risk):,} members at high risk for ALL 3 diseases!")
    print("\nTop 5 highest combined risk:")
    triple_risk['combined_risk'] = (
        triple_risk['oncology_likelihood'] + 
        triple_risk['cardiovascular_likelihood'] + 
        triple_risk['mental_health_likelihood']
    ) / 3
    print(triple_risk.nlargest(5, 'combined_risk')[
        ['member_id', 'age', 'oncology_likelihood', 'cardiovascular_likelihood', 
         'mental_health_likelihood', 'combined_risk']
    ].to_string(index=False))

# %%
# Age-based disease risk analysis
print("\n" + "="*60)
print("DISEASE RISK BY AGE GROUP")
print("="*60)

# Create age groups
risk_report['age_group'] = pd.cut(
    risk_report['age'], 
    bins=[0, 30, 40, 50, 60, 70, 100],
    labels=['<30', '30-39', '40-49', '50-59', '60-69', '70+']
)

age_disease_risk = risk_report.groupby('age_group').agg({
    'oncology_likelihood': 'mean',
    'cardiovascular_likelihood': 'mean',
    'mental_health_likelihood': 'mean',
    'member_id': 'count'
}).rename(columns={'member_id': 'member_count'})

print("\nAverage Disease Likelihood by Age Group:")
print(age_disease_risk.to_string())

# Visualize
age_disease_risk_plot = age_disease_risk[
    ['oncology_likelihood', 'cardiovascular_likelihood', 'mental_health_likelihood']
]

age_disease_risk_plot.plot(kind='bar', figsize=(12, 6))
plt.title('Average Disease Likelihood by Age Group', fontweight='bold', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Average Likelihood')
plt.legend(['Oncology', 'Cardiovascular', 'Mental Health'])
plt.xticks(rotation=0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Save Predictions

# %%
print("="*60)
print("RISK DISTRIBUTION ANALYSIS")
print("="*60)

total_members = len(risk_report)

print(f"\nTotal Members Analyzed: {total_members:,}")
print(f"\nRisk Category Distribution:")
risk_dist = risk_report['risk_category'].value_counts().sort_index()
print(risk_dist)

print(f"\nPercentage Distribution:")
risk_pct = (risk_report['risk_category'].value_counts(normalize=True) * 100).sort_index()
print(risk_pct.round(2))

# Visualize risk distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
risk_dist.plot(kind='bar', ax=axes[0], color=['green', 'yellow', 'orange', 'red'])
axes[0].set_title('Risk Category Distribution', fontweight='bold')
axes[0].set_xlabel('Risk Category')
axes[0].set_ylabel('Number of Members')
axes[0].tick_params(axis='x', rotation=0)

# Pie chart
risk_dist.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
               colors=['green', 'yellow', 'orange', 'red'])
axes[1].set_title('Risk Category Proportion', fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# %%
# High risk members analysis
high_risk = risk_report[risk_report['risk_category'].isin(['High', 'Very High'])]

