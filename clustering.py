import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px

from resources.platnets_constants import inputfile

# === STEP 1: Load and clean the dataset ===
df = pd.read_csv(inputfile)

# Select relevant features for embedding
features = [
    "koi_prad",     # Planetary radius
    "koi_period",   # Orbital period
    "koi_duration", # Transit duration
    "koi_depth",    # Transit depth
    "koi_teq",      # Equilibrium temperature
    "koi_insol",    # Insolation flux
    "koi_steff",    # Stellar temperature
    "koi_slogg",    # Stellar surface gravity
    "koi_srad",     # Stellar radius
    "koi_score"     # Confidence score
]

# Ensure the dataset has necessary columns
required_cols = features + ["kepoi_name", "koi_disposition", "ra", "dec"]
df_clean = df[required_cols].dropna()

# === STEP 2: Scale the data ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# === STEP 3: t-SNE for dimensionality reduction ===
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# === STEP 4: K-Means clustering ===
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_tsne)

# === STEP 5: Add results back to DataFrame ===
df_clean["tsne_x"] = X_tsne[:, 0]
df_clean["tsne_y"] = X_tsne[:, 1]
df_clean["cluster"] = clusters

# === STEP 6: Save results ===
df_clean.to_csv("kepler_tsne_clusters.csv", index=False)
print("✅ Saved clustered data to 'kepler_tsne_clusters.csv'")

# === STEP 7: Interactive 2D t-SNE Plot ===
fig_tsne = px.scatter(
    df_clean, x="tsne_x", y="tsne_y",
    color="cluster",
    hover_data=["kepoi_name", "koi_prad", "koi_teq", "koi_disposition"],
    title="Kepler Exoplanets – t-SNE + K-Means Clustering",
    labels={"cluster": "Cluster"}
)
fig_tsne.show()

# === STEP 8: Sky Globe Plot (RA/Dec) ===
fig_globe = px.scatter_geo(
    df_clean,
    lon="ra",
    lat="dec",
    color="cluster",
    projection="orthographic",
    title="Sky Globe of Kepler Exoplanets (RA/Dec by Cluster)",
    hover_name="kepoi_name",
    hover_data=["koi_prad", "koi_teq", "koi_disposition"]
)

fig_globe.update_geos(
    showcoastlines=False,
    showland=False,
    showocean=False,
    lataxis_showgrid=True,
    lonaxis_showgrid=True
)
fig_globe.show()
