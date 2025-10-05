import pickle
import streamlit as st
import pandas as pd
import numpy as np
from os.path import exists
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from resources.platnets_constants import (
    inputfile, df_clean_pkl, y_values_pkl, new_df_clean_pkl, clf_pkl,
    X_test_pkl, y_test_pkl, le_pkl, clf_k2_pkl, le_k2_pkl, k2_pandc_input_file, toi_input_file, clf_toi_pkl, le_toi_pkl
)

# === Helper: Convert RA/DEC to 3D Coordinates ===
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

# Page Configuration
st.set_page_config(page_title="Exoplanet Explorer", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; font-size: 3em;'>🪐 A World Away: Hunting for Exoplanets with AI</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- OVERVIEW SECTION ---
st.subheader("🔭 Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Kepler Objects of Interest (KOI)")
    st.markdown("- NASA’s Kepler mission discovered thousands of planet candidates.")
    st.markdown("- Focuses on transit photometry.")
    st.markdown("- Rich dataset for training AI classifiers.")

with col2:
    st.markdown("### TESS Objects of Interest (TOI)")
    st.markdown("- Transiting Exoplanet Survey Satellite (TESS) scans entire sky.")
    st.markdown("- Short-period exoplanets around bright stars.")
    st.markdown("- Constantly updating catalog.")

with col3:
    st.markdown("### K2 Planets and Candidates")
    st.markdown("- K2 is the extended Kepler mission.")
    st.markdown("- Targets different fields along the ecliptic.")
    st.markdown("- Useful for diverse star systems and noise conditions.")

st.markdown("---")

# --- KOI ANALYSIS ---
st.subheader("🧪 KOI (Kepler) AI Analysis")

y = read_pickle(y_values_pkl)
df_clean = read_pickle(new_df_clean_pkl)
le = read_pickle(le_pkl)
clf = read_pickle(clf_pkl)
X_test = read_pickle(X_test_pkl)
y_test = read_pickle(y_test_pkl)

y_pred = clf.predict(X_test)

st.write(f"**Model Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)).transpose()
st.dataframe(report_df.style.format("{:.2f}"))

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
fig_cm.update_layout(title_text='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual', yaxis_autorange='reversed')
st.plotly_chart(fig_cm, use_container_width=True)

df = pd.read_csv(inputfile)
features = [
    "koi_score", "koi_period", "koi_period_err1", "koi_period_err2",
    "koi_duration", "koi_duration_err1", "koi_duration_err2",
    "koi_depth", "koi_depth_err1", "koi_depth_err2",
    "koi_prad", "koi_prad_err1", "koi_prad_err2",
    "koi_teq", "koi_teq_err1", "koi_teq_err2",
    "koi_insol", "koi_insol_err1", "koi_insol_err2",
    "koi_model_snr", "koi_steff", "koi_steff_err1", "koi_steff_err2",
    "koi_slogg", "koi_slogg_err1", "koi_slogg_err2",
    "koi_srad", "koi_srad_err1", "koi_srad_err2"
]
target = "koi_disposition"
df_filtered = df[features + [target, "kepoi_name", "ra", "dec"]]
missing_ratios = df_filtered.isna().mean()
df_filtered = df_filtered.loc[:, missing_ratios < 0.5]
features = [f for f in features if f in df_filtered.columns]
df_clean = df_filtered.dropna()

importances = clf.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=True)

fig_feat = go.Figure(go.Bar(
    x=feat_importance.values,
    y=feat_importance.index,
    orientation='h',
    marker=dict(color='skyblue')
))
fig_feat.update_layout(title="Feature Importances", xaxis_title="Importance", yaxis_title="Feature")
st.plotly_chart(fig_feat, use_container_width=True)

df_vis = df.dropna(subset=["ra", "dec", "koi_prad", "koi_disposition", "kepoi_name"])
df_vis['disposition_code'] = le.transform(df_vis['koi_disposition'])
df_vis['x'], df_vis['y'], df_vis['z'] = ra_dec_to_xyz(df_vis['ra'], df_vis['dec'])
marker_sizes = (df_vis['koi_prad'] / df_vis['koi_prad'].max()) * 20 + 5

scatter = go.Scatter3d(
    x=df_vis['x'], y=df_vis['y'], z=df_vis['z'],
    mode='markers',
    marker=dict(size=marker_sizes, color=df_vis['disposition_code'], colorscale='Viridis', opacity=0.8,
                colorbar=dict(title="Disposition"), line=dict(width=0.5, color='DarkSlateGrey')),
    text=df_vis.apply(lambda row: f"KOI Name: {row['kepoi_name']}<br>Planet Radius: {row['koi_prad']:.2f} Earth radii<br>Disposition: {row['koi_disposition']}", axis=1),
    hoverinfo='text'
)
fig_3d = go.Figure(data=[scatter])
fig_3d.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='cube',
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
                     title="Kepler Exoplanets on Celestial Globe (RA/Dec)", margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_3d, use_container_width=True)

# --- K2 ANALYSIS ---
st.markdown("### 🧪 K2 Planets and Candidates")

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

st.write(f"**K2 Model Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
report_df = pd.DataFrame(classification_report(y_test, y_pred, target_names=list(le.classes_), zero_division=0, output_dict=True)).transpose()
st.dataframe(report_df.style.format("{:.2f}"))

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
st.plotly_chart(fig_cm, use_container_width=True)

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
st.plotly_chart(fig_feat, use_container_width=True)

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

st.plotly_chart(fig, use_container_width=True)

# --- TOI ANALYSIS ---
st.markdown("### 🧪 TESS Objects of Interest (TOI)")

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

st.write(f"**TESS Model Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
report_df = pd.DataFrame(classification_report(
    y_test, y_pred, target_names=list(le.classes_), zero_division=0, output_dict=True
)).transpose()
st.dataframe(report_df.style.format("{:.2f}"))

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
st.plotly_chart(fig_cm, use_container_width=True)

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
st.plotly_chart(fig_feat, use_container_width=True)

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
st.plotly_chart(fig, use_container_width=True)

# === 🌌 Celestial Globe: All Known Exoplanets (Attractive Interactive Section) ===
st.markdown("## 🌍 Celestial Globe: Explore All Known Exoplanets")

# Load datasets
df_koi = pd.read_csv(inputfile)
df_k2 = pd.read_csv(k2_pandc_input_file)
df_toi = pd.read_csv(toi_input_file)

# Prepare KOI data
koi_points = df_koi.dropna(subset=["ra", "dec", "koi_prad"]).copy()
koi_points["radius"] = koi_points["koi_prad"]
koi_points["source"] = "Kepler (KOI)"

# Prepare K2 data
k2_points = df_k2.dropna(subset=["ra", "dec", "pl_rade"]).copy()
k2_points["radius"] = k2_points["pl_rade"]
k2_points["source"] = "K2"

# Prepare TOI data
toi_points = df_toi.dropna(subset=["ra", "dec", "pl_rade"]).copy()
toi_points["radius"] = toi_points["pl_rade"]
toi_points["source"] = "TESS (TOI)"

# Merge all datasets
df_all = pd.concat(
    [
        koi_points[["ra", "dec", "radius", "source"]],
        k2_points[["ra", "dec", "radius", "source"]],
        toi_points[["ra", "dec", "radius", "source"]],
    ],
    ignore_index=True
)

# Convert RA/DEC to lon/lat for celestial plotting
df_all["lon"] = df_all["ra"]
df_all["lat"] = df_all["dec"]

# Streamlit interactive controls
selected_datasets = st.multiselect(
    "🪐 Select Dataset(s) to Display:",
    ["Kepler (KOI)", "K2", "TESS (TOI)"],
    default=["Kepler (KOI)", "K2", "TESS (TOI)"]
)
rot_lon = st.slider("🌐 Rotate Longitude", 0, 360, 0)
rot_lat = st.slider("📐 Rotate Latitude", -90, 90, 0)

# Filter by selection
df_filtered = df_all[df_all["source"].isin(selected_datasets)]

# Define distinct colors for datasets
dataset_colors = {
    "Kepler (KOI)": "gold",
    "K2": "deeppink",
    "TESS (TOI)": "limegreen"
}

# Build figure
fig_globe = go.Figure()

for src, group in df_filtered.groupby("source"):
    fig_globe.add_trace(go.Scattergeo(
        lon=group["lon"],
        lat=group["lat"],
        text=[
            f"{src}<br>RA: {ra:.2f}°<br>Dec: {dec:.2f}°<br>Radius: {r:.2f} Earth radii"
            for ra, dec, r in zip(group["lon"], group["lat"], group["radius"])
        ],
        name=src,
        mode="markers",
        marker=dict(
            size=np.clip(group["radius"], 3, 12),
            color=dataset_colors.get(src, "white"),
            opacity=0.8,
            line=dict(width=0.5, color="black"),
        ),
        hoverinfo="text"
    ))

# Update layout for a "space" look and centered legend
fig_globe.update_layout(
    geo=dict(
        projection=dict(type="orthographic", rotation=dict(lon=rot_lon, lat=rot_lat)),
        showland=True,
        landcolor="rgb(10,10,25)",
        showocean=True,
        oceancolor="rgb(5,5,50)",
        showcountries=False,
        showcoastlines=False,
        showframe=False,
        bgcolor="black",
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    margin=dict(l=0, r=0, t=0, b=0),
    legend=dict(
        orientation="h",
        x=0.5,
        y=-0.1,
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12)
    ),
)

# Display the Plotly globe
st.plotly_chart(fig_globe, use_container_width=True)
