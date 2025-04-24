import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import shap
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# === Setup ===
analysis_dir = "analysis_runs"
summary_path = os.path.join(analysis_dir, "summary.csv")
output_dir = os.path.join(analysis_dir, "plots_extended")
os.makedirs(output_dir, exist_ok=True)

# === Load Data ===
df = pd.read_csv(summary_path)
print("Duplicate columns:", df.columns[df.columns.duplicated()].tolist())
numeric_cols = [
    'avg_loss', 'val_loss', 'perplexity', 'token_accuracy',
    'learning_rate', 'embed_size', 'k', 'chunk_size',
    'inner_layers', 'block_size', 'batch_size', 'tinystories_weight'
]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# === Plot 1: 3D Interaction - embed_size x inner_layers ===
if {'embed_size', 'inner_layers', 'val_loss'}.issubset(df.columns):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    subset = df.dropna(subset=['embed_size', 'inner_layers', 'val_loss'])
    ax.scatter(subset['embed_size'], subset['inner_layers'], subset['val_loss'], c='r', marker='o')
    ax.set_xlabel('Embed Size')
    ax.set_ylabel('Inner Layers')
    ax.set_zlabel('Val Loss')
    ax.set_title('Val Loss: Embed Size vs Inner Layers')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "3d_embed_innerlayers_valloss.png"))
    plt.close(fig)

# === Plot 2: TSNE on Config Vectors ===
config_data = df[numeric_cols].fillna(0)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(config_data)
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)
df['tsne-2d-one'] = tsne_result[:, 0]
df['tsne-2d-two'] = tsne_result[:, 1]
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(
    x='tsne-2d-one', y='tsne-2d-two', hue='model_type', data=df, ax=ax, palette='Set2'
)
ax.set_title("t-SNE Projection of Config Space")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "tsne_config_space.png"))
plt.close(fig)

# === Plot 3: Clustering by Performance Metrics ===
performance = df[['val_loss', 'token_accuracy', 'perplexity']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42).fit(performance)
df['cluster'] = kmeans.labels_
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(data=df, x='val_loss', y='token_accuracy', hue='cluster', palette='tab10', ax=ax)
ax.set_title("Clustered Models by Performance")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "clustered_by_performance.png"))
plt.close(fig)

# === Plot 4: Parameter Efficiency ===
df['param_efficiency'] = df['val_loss'] / (df['embed_size'] * df['block_size'] * df['inner_layers']).replace(0, np.nan)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='token_accuracy', y='param_efficiency', hue='model_type', style='activation', ax=ax)
ax.set_title("Parameter Efficiency vs Token Accuracy")
ax.set_yscale('log')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, "param_efficiency_vs_token_accuracy.png"))
plt.close(fig)

# === Plot 5: SHAP Hyperparameter Importance ===
# Keep only one 'val_loss' column if duplicates exist
df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate columns across the whole df

df_shap = df[numeric_cols + ['val_loss']].dropna()
df_shap = df_shap.select_dtypes(include=[np.number])

# Confirm val_loss is present and clean
y = df_shap['val_loss']
if isinstance(y, pd.DataFrame):
    y = y.iloc[:, 0]  # Convert to Series if still a DataFrame

X = df_shap.drop(columns=['val_loss']).astype(np.float32)
y = y.astype(np.float32)

assert y.ndim == 1, "y should be a 1D Series"

model = xgb.XGBRegressor(tree_method="hist")
model.fit(X, y)

explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP Summary Plot: Hyperparameter Importance")
plt.savefig(os.path.join(output_dir, "shap_hyperparam_importance.png"), bbox_inches='tight')
plt.close()
