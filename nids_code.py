# ============================================
# NETWORK INTRUSION DETECTION SYSTEM (NIDS)
# Using Machine Learning
# Based on NSL-KDD Dataset
# ============================================

# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')

# For Google Colab mounting (if using Colab)
# from google.colab import drive
# drive.mount('/content/drive')

print("Libraries imported successfully!")


# ============================================
# 2. DEFINE COLUMN NAMES FOR NSL-KDD DATASET
# ============================================

# NSL-KDD has 41 features + 1 label
col_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

print(f"Total columns: {len(col_names)}")



# ============================================
# 3. LOAD DATASET
# ============================================

# Option 1: Download from URL (NSL-KDD)
def load_nsl_kdd():
    """
    Load NSL-KDD dataset from online source
    """
    print("Downloading NSL-KDD dataset...")
    
    # URLs for NSL-KDD dataset
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    
    # Load training data
    train_data = pd.read_csv(train_url, header=None, names=col_names)
    print(f"Training data loaded: {train_data.shape}")
    
    # Load testing data
    test_data = pd.read_csv(test_url, header=None, names=col_names)
    print(f"Testing data loaded: {test_data.shape}")
    
    return train_data, test_data

# Option 2: If you have local files
# train_data = pd.read_csv('KDDTrain+.txt', header=None, names=col_names)
# test_data = pd.read_csv('KDDTest+.txt', header=None, names=col_names)

# Load the data
train_data, test_data = load_nsl_kdd()

# Display first few rows
print("\nFirst 5 rows of training data:")
train_data.head()


# ============================================
# 4. DATA EXPLORATION
# ============================================

print("=" * 50)
print("DATA EXPLORATION")
print("=" * 50)

# Dataset shape
print(f"\nTraining data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

# Check for missing values
print(f"\nMissing values in training data: {train_data.isnull().sum().sum()}")
print(f"Missing values in testing data: {test_data.isnull().sum().sum()}")

# Data types
print(f"\nData types:\n{train_data.dtypes.value_counts()}")

# Basic statistics
print(f"\nBasic statistics of numeric columns:")
train_data.describe()



# ============================================
# 5. DATA PREPROCESSING
# ============================================

def preprocess_data(data):
    """
    Preprocess the data according to the documentation:
    - Remove 'difficulty_level' column
    - Handle categorical variables
    - Convert labels to attack categories
    """
    
    df = data.copy()
    
    # Remove 'difficulty_level' as per documentation (page 29)
    if 'difficulty_level' in df.columns:
        df.drop(['difficulty_level'], axis=1, inplace=True)
        print("Removed 'difficulty_level' column")
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Encoded {col}")
    
    return df

# Preprocess training and testing data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

print(f"\nTraining data shape after preprocessing: {train_data.shape}")
print(f"Testing data shape after preprocessing: {test_data.shape}")



# ============================================
# 6. CONVERT ATTACK LABELS TO CATEGORIES
# ============================================

def map_attack_categories(df):
    """
    Map individual attack types to main categories:
    - Normal
    - DoS (Denial of Service)
    - Probe
    - R2L (Remote to Local)
    - U2R (User to Root)
    """
    
    df = df.copy()
    
    # DoS attacks
    dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
                   'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    
    # R2L attacks
    r2l_attacks = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 
                   'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 
                   'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
    
    # Probe attacks
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    
    # U2R attacks
    u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 
                   'sqlattack', 'xterm']
    
    # Create category column
    def get_category(label):
        if label == 'normal':
            return 'normal'
        elif label in dos_attacks:
            return 'dos'
        elif label in probe_attacks:
            return 'probe'
        elif label in r2l_attacks:
            return 'r2l'
        elif label in u2r_attacks:
            return 'u2r'
        else:
            return 'unknown'
    
    df['attack_category'] = df['label'].apply(get_category)
    
    # Create binary label (normal vs attack)
    df['is_attack'] = (df['attack_category'] != 'normal').astype(int)
    
    print("Attack categories mapped successfully!")
    print(f"\nDistribution of attack categories:")
    print(df['attack_category'].value_counts())
    
    return df

# Apply mapping
train_data = map_attack_categories(train_data)
test_data = map_attack_categories(test_data)



# ============================================
# 7. FEATURE SELECTION (Correlation Analysis)
# ============================================

def select_features_by_correlation(df, target_col, threshold=0.5):
    """
    Select features with correlation above threshold (as per documentation page 32)
    """
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation with target
    correlations = df[numeric_cols].corrwith(df[target_col]).abs()
    
    # Select features above threshold
    selected_features = correlations[correlations > threshold].index.tolist()
    
    # Remove target from selected features
    if target_col in selected_features:
        selected_features.remove(target_col)
    
    print(f"Features with correlation > {threshold}: {len(selected_features)}")
    print(f"Selected features: {selected_features[:10]}..." if len(selected_features) > 10 else f"Selected features: {selected_features}")
    
    return selected_features

# Select features for binary classification
binary_features = select_features_by_correlation(train_data, 'is_attack', threshold=0.3)
print(f"\nBinary classification features: {binary_features}")



# ============================================
# 8. DATA NORMALIZATION (StandardScaler)
# ============================================

def normalize_data(df, features):
    """
    Normalize data using StandardScaler (as per documentation page 30)
    """
    
    df_normalized = df.copy()
    
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    df_normalized[features] = scaler.fit_transform(df[features])
    
    print("Data normalized using StandardScaler")
    
    return df_normalized, scaler

# Prepare features for binary classification
binary_features_clean = [f for f in binary_features if f in train_data.columns]

# Normalize the data
train_data_normalized, scaler = normalize_data(train_data, binary_features_clean)

# Show normalized data
print("\nNormalized data sample:")
train_data_normalized[binary_features_clean[:5]].head()




# ============================================
# 9. PREPARE DATA FOR MODEL TRAINING
# ============================================

def prepare_data_for_training(df, features, target_col):
    """
    Prepare X and y for model training
    """
    
    # Ensure all features exist
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].values
    y = df[target_col].values
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features used: {len(available_features)}")
    
    return X, y, available_features

# Prepare for BINARY classification (Normal vs Attack)
X_binary, y_binary, binary_used_features = prepare_data_for_training(
    train_data_normalized, binary_features_clean, 'is_attack'
)

# Prepare for MULTI-CLASS classification (Attack types)
X_multi, y_multi, _ = prepare_data_for_training(
    train_data_normalized, binary_features_clean, 'attack_category'
)

# Encode multi-class labels
label_encoder_multi = LabelEncoder()
y_multi_encoded = label_encoder_multi.fit_transform(y_multi)

print(f"\nMulti-class labels: {label_encoder_multi.classes_}")



# ============================================
# 10. SPLIT DATA (75% training, 25% testing)
# ============================================

# Split for binary classification
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.25, random_state=42
)

# Split for multi-class classification
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi_encoded, test_size=0.25, random_state=42
)

print("=" * 50)
print("DATA SPLIT COMPLETE")
print("=" * 50)
print(f"Binary - Train: {X_train_bin.shape}, Test: {X_test_bin.shape}")
print(f"Multi  - Train: {X_train_multi.shape}, Test: {X_test_multi.shape}")




# ============================================
# 11. TRAIN SVM MODEL (Binary Classification)
# ============================================

print("=" * 50)
print("TRAINING SVM FOR BINARY CLASSIFICATION")
print("=" * 50)

# Initialize SVM with linear kernel (as per documentation page 35)
svm_binary = SVC(kernel='linear', random_state=42, gamma='auto')

# Train the model
svm_binary.fit(X_train_bin, y_train_bin)
print("SVM model trained successfully!")

# Make predictions
y_pred_bin = svm_binary.predict(X_test_bin)

# Evaluate
print("\n" + "=" * 50)
print("BINARY CLASSIFICATION RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
print(f"Precision: {precision_score(y_test_bin, y_pred_bin):.4f}")
print(f"Recall: {recall_score(y_test_bin, y_pred_bin):.4f}")
print(f"F1-Score: {f1_score(y_test_bin, y_pred_bin):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred_bin, 
                            target_names=['Normal', 'Attack']))

# Confusion Matrix
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
print("\nConfusion Matrix:")
print(cm_bin)



# ============================================
# 12. VISUALIZE BINARY CLASSIFICATION RESULTS
# ============================================

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.title('Confusion Matrix - Binary Classification (Normal vs Attack)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Create a bar chart of metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [
    accuracy_score(y_test_bin, y_pred_bin),
    precision_score(y_test_bin, y_pred_bin),
    recall_score(y_test_bin, y_pred_bin),
    f1_score(y_test_bin, y_pred_bin)
]

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Binary Classification Performance Metrics')
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom')
plt.show()





# ============================================
# 13. TRAIN SVM MODEL (Multi-Class Classification)
# ============================================

print("=" * 50)
print("TRAINING SVM FOR MULTI-CLASS CLASSIFICATION")
print("=" * 50)

# Initialize SVM with linear kernel for multi-class
svm_multi = SVC(kernel='linear', random_state=42, gamma='auto', decision_function_shape='ovr')

# Train the model
svm_multi.fit(X_train_multi, y_train_multi)
print("Multi-class SVM model trained successfully!")

# Make predictions
y_pred_multi = svm_multi.predict(X_test_multi)

# Evaluate
print("\n" + "=" * 50)
print("MULTI-CLASS CLASSIFICATION RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi):.4f}")

print("\nClassification Report:")
print(classification_report(y_test_multi, y_pred_multi, 
                            target_names=label_encoder_multi.classes_))

# Confusion Matrix for multi-class
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
print("\nConfusion Matrix:")
print(cm_multi)



# ============================================
# 14. VISUALIZE MULTI-CLASS RESULTS
# ============================================

# Plot confusion matrix for multi-class
plt.figure(figsize=(10, 8))
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='RdYlGn',
            xticklabels=label_encoder_multi.classes_,
            yticklabels=label_encoder_multi.classes_)
plt.title('Confusion Matrix - Multi-Class Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Create comparison bar chart for multi-class
# Get precision, recall, f1 for each class
from sklearn.metrics import precision_recall_fscore_support

precision_multi, recall_multi, f1_multi, _ = precision_recall_fscore_support(
    y_test_multi, y_pred_multi, average=None
)

classes = label_encoder_multi.classes_
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, precision_multi, width, label='Precision')
bars2 = ax.bar(x, recall_multi, width, label='Recall')
bars3 = ax.bar(x + width, f1_multi, width, label='F1-Score')

ax.set_ylabel('Score')
ax.set_title('Multi-Class Classification Metrics by Attack Type')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()





# ============================================
# 15. COMPARE WITH OTHER ALGORITHMS
# ============================================

def compare_algorithms(X_train, X_test, y_train, y_test, problem_type="binary"):
    """
    Compare multiple ML algorithms (as per documentation page 19)
    """
    
    algorithms = {
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    
    print("\n" + "=" * 50)
    print(f"ALGORITHM COMPARISON ({problem_type.upper()})")
    print("=" * 50)
    
    for name, model in algorithms.items():
        # Train
        model.fit(X_train, y_train)
        # Predict
        y_pred = model.predict(X_test)
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        
        results.append({
            'Algorithm': name,
            'Accuracy': acc
        })
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.4f}")
        
        if problem_type == "binary":
            print(f"  Precision: {precision_score(y_test, y_pred, average='binary'):.4f}")
            print(f"  Recall: {recall_score(y_test, y_pred, average='binary'):.4f}")
    
    return pd.DataFrame(results)

# Compare algorithms for binary classification
comparison_df = compare_algorithms(X_train_bin, X_test_bin, y_train_bin, y_test_bin, "binary")

# Plot comparison
plt.figure(figsize=(10, 6))
bars = plt.bar(comparison_df['Algorithm'], comparison_df['Accuracy'], 
               color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Algorithm Comparison for Binary Classification')
for bar, val in zip(bars, comparison_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom')
plt.show()



# ============================================
# 16. ATTACK DISTRIBUTION VISUALIZATION
# ============================================

# Distribution of attack types (as per documentation page 30)
attack_dist = train_data['attack_category'].value_counts()

plt.figure(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
bars = plt.bar(attack_dist.index, attack_dist.values, color=colors)

plt.title('Distribution of Network Traffic Types', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Number of Instances', fontsize=12)

for bar, val in zip(bars, attack_dist.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
             f'{val:,}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nAttack Distribution:")
print(attack_dist)
print(f"\nPercentage of Attacks: {(attack_dist['dos'] + attack_dist['probe'] + attack_dist['r2l'] + attack_dist['u2r']) / attack_dist.sum() * 100:.2f}%")




# ============================================
# 17. FEATURE CORRELATION HEATMAP
# ============================================

# Create correlation heatmap for top features
plt.figure(figsize=(14, 12))

# Select top 15 features for visualization
top_features = binary_features_clean[:15] if len(binary_features_clean) > 15 else binary_features_clean
corr_matrix = train_data_normalized[top_features + ['is_attack']].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap (Top Features vs Attack Label)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




# ============================================
# 18. CROSS-VALIDATION (Robustness Testing)
# ============================================

print("=" * 50)
print("CROSS-VALIDATION RESULTS")
print("=" * 50)

# Perform 5-fold cross-validation for binary model
cv_scores = cross_val_score(svm_binary, X_binary, y_binary, cv=5, scoring='accuracy')

print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# For multi-class model
cv_scores_multi = cross_val_score(svm_multi, X_multi, y_multi_encoded, cv=5, scoring='accuracy')
print(f"\nMulti-Class 5-Fold CV Scores: {cv_scores_multi}")
print(f"Mean CV Accuracy: {cv_scores_multi.mean():.4f} (+/- {cv_scores_multi.std() * 2:.4f})")


# ============================================
# 19. FINAL SUMMARY REPORT
# ============================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("""
╔══════════════════════════════════════════════════════════════════╗
║     NETWORK INTRUSION DETECTION SYSTEM (NIDS)                    ║
║     Machine Learning Approach                                    ║
╚══════════════════════════════════════════════════════════════════╝

DATASET INFORMATION:
├── Dataset: NSL-KDD (Improved KDD Cup 1999)
├── Training samples: 125,973
├── Testing samples: 22,544
├── Features: 41
└── Attack categories: DoS, Probe, R2L, U2R

PREPROCESSING STEPS:
├── Removed 'difficulty_level' column
├── Encoded categorical variables (protocol_type, service, flag)
├── Mapped attack labels to categories
├── Feature selection via correlation analysis
└── Normalization using StandardScaler

BINARY CLASSIFICATION RESULTS (Normal vs Attack):
├── Algorithm: SVM (Linear Kernel)
├── Accuracy: {:.4f}
├── Precision: {:.4f}
├── Recall: {:.4f}
└── F1-Score: {:.4f}

MULTI-CLASS CLASSIFICATION RESULTS:
├── Algorithm: SVM (Linear Kernel)
├── Accuracy: {:.4f}
└── Classes: {}

CROSS-VALIDATION:
├── Binary CV Mean Accuracy: {:.4f}
└── Multi-Class CV Mean Accuracy: {:.4f}

""".format(
    accuracy_score(y_test_bin, y_pred_bin),
    precision_score(y_test_bin, y_pred_bin),
    recall_score(y_test_bin, y_pred_bin),
    f1_score(y_test_bin, y_pred_bin),
    accuracy_score(y_test_multi, y_pred_multi),
    list(label_encoder_multi.classes_),
    cv_scores.mean(),
    cv_scores_multi.mean()
))

print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)



# ============================================
# 20. SAVE MODEL (Optional - for deployment)
# ============================================

import joblib

# Save the trained models
joblib.dump(svm_binary, 'nids_binary_model.pkl')
joblib.dump(svm_multi, 'nids_multi_model.pkl')
joblib.dump(scaler, 'nids_scaler.pkl')
joblib.dump(label_encoder_multi, 'nids_label_encoder.pkl')

print("\nModels saved successfully!")
print("- nids_binary_model.pkl")
print("- nids_multi_model.pkl")
print("- nids_scaler.pkl")
print("- nids_label_encoder.pkl")
