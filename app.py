import pickle
from os.path import exists
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from resources.platnets_constants import (
    inputfile, df_clean_pkl, y_values_pkl, new_df_clean_pkl, clf_pkl,
    X_test_pkl, y_test_pkl, le_pkl
)
from pandas import Index

def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)

# === Helper: Convert RA/DEC to 3D Coordinates ===
def ra_dec_to_xyz(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return x, y, z


# === STEP 1: Load and Clean Dataset ===
df = pd.read_csv(inputfile)

features = [
    "koi_score",
    "koi_period", "koi_period_err1", "koi_period_err2",
    "koi_duration", "koi_duration_err1", "koi_duration_err2",
    "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_prad", "koi_prad_err1", "koi_prad_err2",
    "koi_teq", "koi_teq_err1", "koi_teq_err2",
    "koi_insol", "koi_insol_err1", "koi_insol_err2",
    "koi_model_snr",
    "koi_steff", "koi_steff_err1", "koi_steff_err2",
    "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
    "koi_srad", "koi_srad_err1", "koi_srad_err2"
]
target = "koi_disposition"

required_cols = features + [target, "kepoi_name", "ra", "dec"]

# Keep only required columns, and drop features with too many missing values
df_filtered = df[required_cols]
missing_ratios = df_filtered.isna().mean()
df_filtered = df_filtered.loc[:, missing_ratios < 0.5]  # drop columns >50% missing

# Update feature list accordingly
features = [f for f in features if f in df_filtered.columns]

# Drop rows with any remaining NaNs
df_clean = df_filtered.dropna()

# Save cleaned dataset
if not exists(df_clean_pkl):
    save_as_pickle(df_clean_pkl, df_clean)
else:
    df_clean = read_pickle(df_clean_pkl)


# === STEP 2: Encode Target Variable ===
if not exists(y_values_pkl):
    le = LabelEncoder()
    y = le.fit_transform(df_clean[target])
    save_as_pickle(y_values_pkl, y)
    save_as_pickle(new_df_clean_pkl, df_clean)
    save_as_pickle(le_pkl, le)
else:
    y = read_pickle(y_values_pkl)
    df_clean = read_pickle(new_df_clean_pkl)
    le = read_pickle(le_pkl)

# === STEP 3: Train Classifier ===
if not exists(clf_pkl):
    # Impute missing values just in case (though we dropped NaNs earlier)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df_clean[features])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save model and test data
    save_as_pickle(clf_pkl, clf)
    save_as_pickle(X_test_pkl, X_test)
    save_as_pickle(y_test_pkl, y_test)
else:
    clf = read_pickle(clf_pkl)
    X_test = read_pickle(X_test_pkl)
    y_test = read_pickle(y_test_pkl)


# === STEP 4: Predict and Evaluate ===
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
z = cm
x_labels = list(le.classes_)
y_labels = list(le.classes_)

fig_cm = ff.create_annotated_heatmap(
    z,
    x=x_labels,
    y=y_labels,
    colorscale='Blues',
    showscale=True,
    hoverinfo='z',
    annotation_text=[[str(v) for v in row] for row in z]
)

fig_cm.update_layout(
    title_text='Confusion Matrix',
    xaxis_title='Predicted',
    yaxis_title='Actual',
    yaxis_autorange='reversed'
)
fig_cm.show()


# === STEP 5: Feature Importance Plot ===
importances = clf.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=True)

fig_feat = go.Figure(go.Bar(
    x=feat_importance.values,
    y=feat_importance.index,
    orientation='h',
    marker=dict(color='skyblue')
))

fig_feat.update_layout(
    title="Feature Importances",
    xaxis_title="Importance",
    yaxis_title="Feature"
)
fig_feat.show()


# === STEP 6: 3D Visualization of Planets on RA/DEC Globe ===
df_vis = df.dropna(subset=["ra", "dec", "koi_prad", "koi_disposition", "kepoi_name"])

# Use the same LabelEncoder as used for model
df_vis['disposition_code'] = le.transform(df_vis['koi_disposition'])

df_vis['x'], df_vis['y'], df_vis['z'] = ra_dec_to_xyz(df_vis['ra'], df_vis['dec'])

size_scale = 20
marker_sizes = (df_vis['koi_prad'] / df_vis['koi_prad'].max()) * size_scale + 5

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
    text=df_vis.apply(lambda row: f"KOI Name: {row['kepoi_name']}<br>"
                                  f"Planet Radius: {row['koi_prad']:.2f} Earth radii<br>"
                                  f"Disposition: {row['koi_disposition']}", axis=1),
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
    title="Kepler Exoplanets on Celestial Globe (RA/Dec)",
    margin=dict(l=0, r=0, t=40, b=0)
)
fig.show()
