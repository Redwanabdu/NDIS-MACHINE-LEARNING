# ============================================
# VISUALIZATION GENERATOR FOR NIDS PROJECT
# Creates publication-ready images for GitHub
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Set professional style for all plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

print("=" * 60)
print("NIDS VISUALIZATION GENERATOR")
print("Creating publication-ready images for GitHub")
print("=" * 60)

# ============================================
# 1. LOAD AND PREPARE DATA (Same as your notebook)
# ============================================

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

print("\n[1/6] Loading dataset...")

# URLs for NSL-KDD dataset
train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

train_data = pd.read_csv(train_url, header=None, names=col_names)
test_data = pd.read_csv(test_url, header=None, names=col_names)

print(f"Training data: {train_data.shape}")
print(f"Testing data: {test_data.shape}")

# ============================================
# 2. PREPROCESSING
# ============================================

print("\n[2/6] Preprocessing data...")

# Remove difficulty_level
train_data.drop(['difficulty_level'], axis=1, inplace=True)
test_data.drop(['difficulty_level'], axis=1, inplace=True)

# Encode categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])

# Map attack categories
def map_attack_categories(df):
    dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 
                   'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
    r2l_attacks = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 
                   'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 
                   'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 
                   'sqlattack', 'xterm']
    
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
    df['is_attack'] = (df['attack_category'] != 'normal').astype(int)
    return df

train_data = map_attack_categories(train_data)
test_data = map_attack_categories(test_data)

# Feature selection
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
correlations = train_data[numeric_cols].corrwith(train_data['is_attack']).abs()
selected_features = correlations[correlations > 0.3].index.tolist()
if 'is_attack' in selected_features:
    selected_features.remove('is_attack')

# Normalize
scaler = StandardScaler()
train_data[selected_features] = scaler.fit_transform(train_data[selected_features])
test_data[selected_features] = scaler.transform(test_data[selected_features])

# Prepare data
X_train = train_data[selected_features].values
y_train = train_data['is_attack'].values
X_test = test_data[selected_features].values
y_test = test_data['is_attack'].values

# ============================================
# 3. TRAIN MODEL
# ============================================

print("\n[3/6] Training SVM model...")

svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

print(f"Model accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ============================================
# 4. GENERATE ALL VISUALIZATIONS
# ============================================

print("\n[4/6] Generating visualizations...")

# Create a directory for images
import os
os.makedirs('github_images', exist_ok=True)

# -----------------------------------------------------------------
# FIGURE 1: Confusion Matrix (Binary Classification)
# -----------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'],
            ax=ax1, annot_kws={'size': 14})
ax1.set_title('Confusion Matrix - Binary Classification', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=12)
ax1.set_ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('github_images/confusion_matrix_binary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: confusion_matrix_binary.png")

# -----------------------------------------------------------------
# FIGURE 2: Performance Metrics Bar Chart
# -----------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
bars = ax2.bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('NIDS Performance Metrics - Binary Classification', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('github_images/performance_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: performance_metrics.png")

# -----------------------------------------------------------------
# FIGURE 3: Attack Distribution Pie Chart
# -----------------------------------------------------------------
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
attack_dist = train_data['attack_category'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
wedges, texts, autotexts = ax3a.pie(attack_dist.values, labels=attack_dist.index, 
                                      autopct='%1.1f%%', colors=colors_pie,
                                      explode=[0.02]*len(attack_dist), shadow=True)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
ax3a.set_title('Distribution of Network Traffic Types', fontsize=12, fontweight='bold')

# Bar chart
bars = ax3b.bar(attack_dist.index, attack_dist.values, color=colors_pie, edgecolor='black')
ax3b.set_xlabel('Category', fontsize=12)
ax3b.set_ylabel('Number of Instances', fontsize=12)
ax3b.set_title('Attack Category Distribution', fontsize=12, fontweight='bold')
ax3b.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, attack_dist.values):
    ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500, 
              f'{val:,}', ha='center', va='bottom', fontsize=10)

plt.suptitle('NSL-KDD Dataset Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('github_images/attack_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: attack_distribution.png")

# -----------------------------------------------------------------
# FIGURE 4: ROC Curve
# -----------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
ax4.fill_between(fpr, tpr, alpha=0.2, color='orange')
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax4.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax4.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
ax4.legend(loc="lower right", fontsize=11)
ax4.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('github_images/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: roc_curve.png")

# -----------------------------------------------------------------
# FIGURE 5: Feature Correlation Heatmap
# -----------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(14, 12))
top_features = selected_features[:15] + ['is_attack']
corr_matrix = train_data[top_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax5)
ax5.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
ax5.tick_params(axis='x', rotation=45, labelsize=9)
ax5.tick_params(axis='y', labelsize=9)
plt.tight_layout()
plt.savefig('github_images/feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: feature_correlation.png")

# -----------------------------------------------------------------
# FIGURE 6: Algorithm Comparison
# -----------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

fig6, ax6 = plt.subplots(figsize=(10, 6))

algorithms = {
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in algorithms.items():
    model.fit(X_train, y_train)
    y_pred_temp = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred_temp)

bars = ax6.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'],
               edgecolor='black', linewidth=1.5)
ax6.set_ylim(0, 1)
ax6.set_ylabel('Accuracy', fontsize=12)
ax6.set_title('Algorithm Comparison for NIDS', fontsize=14, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)
for bar, (name, val) in zip(bars, results.items()):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('github_images/algorithm_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: algorithm_comparison.png")

# -----------------------------------------------------------------
# FIGURE 7: Cross-Validation Results
# -----------------------------------------------------------------
from sklearn.model_selection import cross_val_score

fig7, ax7 = plt.subplots(figsize=(8, 6))

cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)
cv_scores_list = cv_scores.tolist()

bars = ax7.bar(range(1, 6), cv_scores_list, color='teal', edgecolor='black', linewidth=1.5)
ax7.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean = {cv_scores.mean():.3f}')
ax7.set_xlabel('Fold', fontsize=12)
ax7.set_ylabel('Accuracy', fontsize=12)
ax7.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax7.set_ylim(0.9, 1.0)
ax7.legend()
ax7.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, cv_scores_list):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('github_images/cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: cross_validation.png")

# -----------------------------------------------------------------
# FIGURE 8: Precision-Recall by Attack Type (Multi-Class)
# -----------------------------------------------------------------
print("\n[5/6] Generating multi-class visualizations...")

# Prepare multi-class data
X_train_multi = train_data[selected_features].values
y_train_multi = train_data['attack_category'].values
X_test_multi = test_data[selected_features].values
y_test_multi = test_data['attack_category'].values

le_multi = LabelEncoder()
y_train_multi_enc = le_multi.fit_transform(y_train_multi)
y_test_multi_enc = le_multi.transform(y_test_multi)

svm_multi = SVC(kernel='linear', random_state=42)
svm_multi.fit(X_train_multi, y_train_multi_enc)
y_pred_multi = svm_multi.predict(X_test_multi)

from sklearn.metrics import precision_recall_fscore_support
precision_multi, recall_multi, f1_multi, _ = precision_recall_fscore_support(
    y_test_multi_enc, y_pred_multi, average=None
)

fig8, ax8 = plt.subplots(figsize=(12, 6))
classes = le_multi.classes_
x = np.arange(len(classes))
width = 0.25

bars1 = ax8.bar(x - width, precision_multi, width, label='Precision', color='#3498db', edgecolor='black')
bars2 = ax8.bar(x, recall_multi, width, label='Recall', color='#2ecc71', edgecolor='black')
bars3 = ax8.bar(x + width, f1_multi, width, label='F1-Score', color='#e74c3c', edgecolor='black')

ax8.set_ylabel('Score', fontsize=12)
ax8.set_title('Multi-Class Classification Metrics by Attack Type', fontsize=14, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(classes)
ax8.legend(loc='upper right')
ax8.set_ylim(0, 1)
ax8.grid(axis='y', alpha=0.3)

for bars, vals in zip([bars1, bars2, bars3], [precision_multi, recall_multi, f1_multi]):
    for bar, val in zip(bars, vals):
        if val > 0:
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('github_images/multiclass_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: multiclass_metrics.png")

# -----------------------------------------------------------------
# FIGURE 9: Multi-Class Confusion Matrix
# -----------------------------------------------------------------
fig9, ax9 = plt.subplots(figsize=(10, 8))
cm_multi = confusion_matrix(y_test_multi_enc, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=classes, yticklabels=classes, ax=ax9, annot_kws={'size': 11})
ax9.set_title('Confusion Matrix - Multi-Class Classification', fontsize=14, fontweight='bold')
ax9.set_xlabel('Predicted Label', fontsize=12)
ax9.set_ylabel('True Label', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('github_images/confusion_matrix_multiclass.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: confusion_matrix_multiclass.png")

# -----------------------------------------------------------------
# FIGURE 10: Summary Dashboard (All Metrics in One Figure)
# -----------------------------------------------------------------
fig10, axes = plt.subplots(2, 2, figsize=(14, 12))
fig10.suptitle('NIDS Project Summary Dashboard', fontsize=16, fontweight='bold')

# Subplot 1: Binary metrics
ax1_dash = axes[0, 0]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
metrics_values = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]
bars = ax1_dash.bar(metrics_names, metrics_values, color='steelblue', edgecolor='black')
ax1_dash.set_ylim(0, 1)
ax1_dash.set_title('Binary Classification', fontweight='bold')
ax1_dash.set_ylabel('Score')
for bar, val in zip(bars, metrics_values):
    ax1_dash.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                  f'{val:.3f}', ha='center', va='bottom')

# Subplot 2: CV scores
ax2_dash = axes[0, 1]
ax2_dash.plot(range(1, 6), cv_scores, 'bo-', linewidth=2, markersize=10)
ax2_dash.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.3f}')
ax2_dash.set_xlabel('Fold')
ax2_dash.set_ylabel('Accuracy')
ax2_dash.set_title('Cross-Validation (5-Fold)', fontweight='bold')
ax2_dash.set_ylim(0.9, 1.0)
ax2_dash.legend()
ax2_dash.grid(True, alpha=0.3)

# Subplot 3: ROC
ax3_dash = axes[1, 0]
ax3_dash.plot(fpr, tpr, 'darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
ax3_dash.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--')
ax3_dash.fill_between(fpr, tpr, alpha=0.2, color='orange')
ax3_dash.set_xlabel('False Positive Rate')
ax3_dash.set_ylabel('True Positive Rate')
ax3_dash.set_title('ROC Curve', fontweight='bold')
ax3_dash.legend()

# Subplot 4: Algorithm comparison
ax4_dash = axes[1, 1]
algo_names = list(results.keys())
algo_values = list(results.values())
colors_algo = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax4_dash.barh(algo_names, algo_values, color=colors_algo, edgecolor='black')
ax4_dash.set_xlim(0, 1)
ax4_dash.set_xlabel('Accuracy')
ax4_dash.set_title('Algorithm Comparison', fontweight='bold')
for bar, val in zip(bars, algo_values):
    ax4_dash.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                  f'{val:.3f}', va='center')

plt.tight_layout()
plt.savefig('github_images/summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: summary_dashboard.png")

# ============================================
# 5. CREATE A MARKDOWN GALLERY FILE
# ============================================

print("\n[6/6] Creating image gallery for README...")

gallery_md = """# NIDS Project Visualizations Gallery

This page contains all visualizations generated from the Network Intrusion Detection System project.

## 1. Confusion Matrix (Binary Classification)
![Confusion Matrix](github_images/confusion_matrix_binary.png)

## 2. Performance Metrics
![Performance Metrics](github_images/performance_metrics.png)

## 3. Attack Distribution
![Attack Distribution](github_images/attack_distribution.png)

## 4. ROC Curve
![ROC Curve](github_images/roc_curve.png)

## 5. Feature Correlation Heatmap
![Feature Correlation](github_images/feature_correlation.png)

## 6. Algorithm Comparison
![Algorithm Comparison](github_images/algorithm_comparison.png)

## 7. Cross-Validation Results
![Cross Validation](github_images/cross_validation.png)

## 8. Multi-Class Classification Metrics
![Multi-Class Metrics](github_images/multiclass_metrics.png)

## 9. Multi-Class Confusion Matrix
![Multi-Class Confusion Matrix](github_images/confusion_matrix_multiclass.png)

## 10. Summary Dashboard
![Summary Dashboard](github_images/summary_dashboard.png)

---

*Generated automatically from NIDS project*
"""

with open('VISUALIZATION_GALLERY.md', 'w') as f:
    f.write(gallery_md)
print("  ✓ Saved: VISUALIZATION_GALLERY.md")

# ============================================
# 6. PRINT SUMMARY
# ============================================

print("\n" + "=" * 60)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 60)
print("""
All images saved in: 'github_images/' folder

Files created:
├── github_images/
│   ├── confusion_matrix_binary.png
│   ├── performance_metrics.png
│   ├── attack_distribution.png
│   ├── roc_curve.png
│   ├── feature_correlation.png
│   ├── algorithm_comparison.png
│   ├── cross_validation.png
│   ├── multiclass_metrics.png
│   ├── confusion_matrix_multiclass.png
│   └── summary_dashboard.png
│
└── VISUALIZATION_GALLERY.md

HOW TO USE:
1. Upload the 'github_images' folder to your GitHub repository
2. Copy the images to your README.md using:
   ![Description](github_images/filename.png)
3. The VISUALIZATION_GALLERY.md can be added as a wiki page
""")

print("=" * 60)
print("✓ Ready to attract professors!")
print("=" * 60)
