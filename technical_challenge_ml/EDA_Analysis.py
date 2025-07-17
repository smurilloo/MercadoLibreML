# ============================================================================
# EXPLORACION DE DATOS
# ============================================================================
import json, os, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Configurar Pandas para no recortar la salida
pd.set_option('display.max_rows', None)

# === LISTA DE COLUMNAS A IGNORAR (aplanadas) ===
_IGNORED_COLUMNS = {
    "id", "url", "secure_url", "thumbnail", "permalink",
    "address_line", "description", "video_id", "catalog_product_id",
    "descriptions", "pictures", "deal_ids", "attributes", "tags",
    "listing_source", "parent_item_id", "coverage_areas",
    "international_delivery_mode", "official_store_id",
    "differential_pricing", "geolocation_latitude", "geolocation_longitude",
    "seller_contact_area_code2", "seller_contact_phone2",
    "seller_contact_webpage", "seller_contact_email",
    "seller_contact_contact", "seller_contact_area_code",
    "seller_contact_other_info", "seller_contact_phone",
    "location_open_hours", "location_neighborhood_name",
    "location_neighborhood_id", "location_longitude", "location_country_name",
    "location_country_id", "location_address_line", "location_latitude",
    "location_zip_code", "location_city_name", "location_city_id",
    "location_state_name", "location_state_id", "shipping_local_pick_up",
    "shipping_methods", "shipping_tags", "shipping_free_shipping",
    "shipping_mode", "shipping_dimensions", "seller_address_comment",
    "seller_address_longitude", "seller_address_id",
    "seller_address_country_name", "seller_address_country_id",
    "seller_address_address_line", "seller_address_latitude",
    "seller_address_search_location_neighborhood_name",
    "seller_address_search_location_neighborhood_id",
    "seller_address_search_location_state_name",
    "seller_address_search_location_state_id",
    "seller_address_search_location_city_name",
    "seller_address_search_location_city_id", "seller_address_zip_code",
    "seller_address_city_name", "seller_address_city_id",
    "seller_address_state_name", "seller_address_state_id"
}

# ---------- 0. Carpeta ----------
os.makedirs("EDA_Files", exist_ok=True)

# ---------- 1. Lectura ----------
path = os.path.join("technical_challenge_ml", "MLA_100k.jsonlines")
with open(path, "r", encoding="utf-8") as f:
    raw = [json.loads(l) for l in f]
raw = [r for r in raw if "condition" in r and r["condition"] in ("new", "used")]

print("=== RESUMEN GENERAL ===")
print("Total de registros:", len(raw))
print("Distribución de condición:")
condition_counts = pd.Series([r["condition"] for r in raw]).value_counts()
print(condition_counts)

# ---------- Pie Chart: Distribución de "condition" ----------
plt.figure(figsize=(6, 6))
plt.pie(
    condition_counts,
    labels=condition_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=["#3498db", "#e74c3c"]
)
plt.title("Distribución de productos por condición", fontsize=14)
plt.savefig("EDA_Files/condition_pie_chart.png", dpi=300)
plt.close()

# ---------- 2. Aplanar ----------
def _flat(d, p="", s="_"):
    items = []
    for k, v in d.items():
        nk = f"{p}{s}{k}" if p else k
        if isinstance(v, dict):
            items.extend(_flat(v, nk, s).items())
        elif isinstance(v, list):
            items.append((nk, str(v)))
        else:
            items.append((nk, v))
    return dict(items)

df_full = pd.DataFrame([_flat(r) for r in raw])
df_eda  = df_full.loc[:, ~df_full.columns.isin(_IGNORED_COLUMNS)]
df_num  = df_eda.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
df_num  = df_num.loc[:, df_num.var() > 0]

# ---------- 3. Guardar CSV ----------
df_num.to_csv("EDA_Files/eda_raw.csv", index=False, encoding="utf-8")
print("✅ Datos originales guardados en 'EDA_Files/eda_raw.csv'")

print("\n=== MUESTRA DE 10 REGISTROS (sin normalizar) ===")
print(df_num.head(10).round(2))

# ---------- 4. Normalización ----------
scaler   = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

# ---------- 5. Selección de 9 variables más relevantes ----------
cov_norm = df_scaled.cov()
top9 = cov_norm.abs().sum().sort_values(ascending=False).head(9).index.tolist()

# ---------- 6. Histogramas de las 9 principales en una sola figura ----------
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(top9):
    sns.histplot(df_num[col], kde=True, ax=axes[i], color="#e74c3c")
    axes[i].set_title(col, fontsize=14)
    axes[i].set_xlabel("")
fig.suptitle("Histogramas de las 9 variables con mayor varianza (escala original)", fontsize=18, weight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("EDA_Files/histogramas_top9.png", dpi=300)
plt.close()

# ---------- 7. Matriz de correlación (normalizada) ----------
corr_norm = df_scaled.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_norm, cmap="coolwarm", square=True, linewidths=.5, annot=True, fmt=".2f")
plt.title("Matriz de Correlación Normalizada", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("EDA_Files/correlation_matrix_normalized.png", dpi=300)
plt.close()

# ---------- 8. Matriz de covarianza (normalizada) ----------
cmap = sns.color_palette("Reds", as_cmap=True)
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(cov_norm, dtype=bool))
sns.heatmap(
    cov_norm,
    mask=mask,
    cmap=cmap,
    square=True,
    linewidths=0.3,
    linecolor='white',
    cbar_kws={
        "label": "Covarianza normalizada",
        "ticks": np.linspace(np.round(cov_norm.min().min(), 2), np.round(cov_norm.max().max(), 2), 5)
    },
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8, "color": "black"}
)
plt.title("Matriz de Covarianza Normalizada", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("EDA_Files/covariance_matrix_normalized.jpg", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 9. Final ----------
print("\n✅ EDA completo – archivos guardados en 'EDA_Files':")
for f in sorted(os.listdir("EDA_Files")):
    print("  -", f)
