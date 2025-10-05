import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.figure_factory as ff
from resources.platnets_constants import toi_input_file, clf_toi_pkl, le_toi_pkl

def ra_dec_to_xyz(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return x, y, z

def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

# Load TOI dataset
df = pd.read_csv(toi_input_file)

features = [
    "st_rad",
    "st_teff",
    "st_dist",
    "st_tmag",
    "pl_trandep",
    "pl_trandurh",
    "pl_orbper",
    "st_pmdec",
    "st_pmra"
]



target = "tfopwg_disp"
required_cols = features + ["toi", target, "ra", "dec", "pl_rade"]

# Filter dataframe to required columns and drop missing
df_clean = df[required_cols].dropna()

# Prepare features and target
X = df_clean[features]
y_raw = df_clean[target]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Save model and label encoder
save_as_pickle(clf_toi_pkl, clf)
save_as_pickle(le_toi_pkl, le)

# Predict and evaluate
y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=list(le.classes_), zero_division=0
))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=list(le.classes_),
    y=list(le.classes_),
    colorscale='Blues',
    showscale=True,
    annotation_text=[[str(v) for v in row] for row in cm],
    hoverinfo='z'
)
fig_cm.update_layout(
    title="Confusion Matrix (TOI Model)",
    xaxis_title="Predicted",
    yaxis_title="Actual",
    yaxis_autorange='reversed'
)
fig_cm.show()

# Feature importance plot
importances = clf.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=True)

fig_feat = go.Figure(go.Bar(
    x=feat_importance.values,
    y=feat_importance.index,
    orientation='h',
    marker=dict(color='mediumturquoise')
))
fig_feat.update_layout(
    title="Feature Importances (TOI Model)",
    xaxis_title="Importance",
    yaxis_title="Feature"
)
fig_feat.show()

# Prepare data for visualization
df_vis = df_clean.copy()
df_vis['disposition_code'] = le.transform(df_vis[target])
df_vis['x'], df_vis['y'], df_vis['z'] = ra_dec_to_xyz(df_vis['ra'], df_vis['dec'])

# Normalize radius for marker size
size_scale = 20
marker_sizes = (df_vis['pl_rade'] / df_vis['pl_rade'].max()) * size_scale + 5

scatter = go.Scatter3d(
    x=df_vis['x'],
    y=df_vis['y'],
    z=df_vis['z'],
    mode='markers',
    marker=dict(
        size=marker_sizes,
        color=df_vis['disposition_code'],
        colorscale='Viridis',
        opacity=0.8,
        colorbar=dict(title="Disposition"),
        line=dict(width=0.5, color='DarkSlateGrey')
    ),
    text=df_vis.apply(lambda row: f"TOI: {row['toi']}<br>"
                                  f"Planet Radius: {row['pl_rade']:.2f} Earth radii<br>"
                                  f"Disposition: {row[target]}", axis=1),
    hoverinfo='text'
)

fig = go.Figure(data=[scatter])
fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    title="TOI Exoplanets on Celestial Globe (RA/Dec)",
    margin=dict(l=0, r=0, t=40, b=0)
)
fig.show()
