import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from resources.platnets_constants import (
    k2_pandc_input_file, clf_k2_pkl, le_k2_pkl
)
import plotly.graph_objects as go
import plotly.figure_factory as ff

def ra_dec_to_xyz(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return x, y, z

def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

# Step 1: Load data
df = pd.read_csv(k2_pandc_input_file)

features = [
    "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
    "pl_rade", "pl_radeerr1", "pl_radeerr2",
    "pl_radj", "pl_radjerr1", "pl_radjerr2",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "st_rad", "st_raderr1", "st_raderr2",
    "st_met", "st_meterr1", "st_meterr2",
    "st_logg", "st_loggerr1", "st_loggerr2",
    "sy_dist", "sy_disterr1", "sy_disterr2",
    "sy_vmag", "sy_vmagerr1", "sy_vmagerr2"
]

target = "disposition"
required_cols = features + ["pl_name", target, "ra", "dec"]

# Step 2: Clean and align
df_clean = df[required_cols].dropna()
X = df_clean[features]
y_raw = df_clean[target]

# Step 3: Encode target properly
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train new model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Save model (optional)
save_as_pickle(clf_k2_pkl, clf)
save_as_pickle(le_k2_pkl, le)

# Step 7: Predictions and Evaluation
y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=list(le.classes_), zero_division=0
))

# Step 8: Confusion Matrix
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
    title="Confusion Matrix",
    xaxis_title="Predicted",
    yaxis_title="Actual",
    yaxis_autorange='reversed'
)
fig_cm.show()

# Step 9: Feature Importances
importances = clf.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=True)

fig_feat = go.Figure(go.Bar(
    x=feat_importance.values,
    y=feat_importance.index,
    orientation='h',
    marker=dict(color='mediumturquoise')
))
fig_feat.update_layout(
    title="Feature Importances (K2 Model)",
    xaxis_title="Importance",
    yaxis_title="Feature"
)
fig_feat.show()

# Prepare data for visualization
df_vis = df.dropna(subset=["ra", "dec", "pl_rade", "disposition", "pl_name"])

# Use the same LabelEncoder used for K2 model
df_vis['disposition_code'] = le.transform(df_vis['disposition'])

# Convert RA/DEC to XYZ
df_vis['x'], df_vis['y'], df_vis['z'] = ra_dec_to_xyz(df_vis['ra'], df_vis['dec'])

# Normalize radius for marker size
size_scale = 20
marker_sizes = (df_vis['pl_rade'] / df_vis['pl_rade'].max()) * size_scale + 5

# Create 3D scatter plot
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
    text=df_vis.apply(lambda row: f"Planet Name: {row['pl_name']}<br>"
                                  f"Planet Radius: {row['pl_rade']:.2f} Earth radii<br>"
                                  f"Disposition: {row['disposition']}", axis=1),
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
    title="K2 Exoplanets on Celestial Globe (RA/Dec)",
    margin=dict(l=0, r=0, t=40, b=0)
)

fig.show()
