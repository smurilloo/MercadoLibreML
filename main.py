from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import os
import uuid
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="technical_challenge_ml/Static"), name="static")

MODEL_PATH = os.path.join("Models_Dir", "gb_pipeline.pkl")
model = joblib.load(MODEL_PATH)

# --- Set of columns to ignore here ---
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
    "seller_address_state_name", "seller_address_state_id", "base_price"
}

def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if new_key in _IGNORED_COLUMNS:
            continue
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)

def drop_ignored_columns(data):
    return [{k: v for k, v in d.items() if k not in _IGNORED_COLUMNS} for d in data]

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("technical_challenge_ml/Static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict-batch")
async def predict_from_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_json(io.StringIO(content.decode("utf-8")), lines=True)

        # Convert datetime columns to string to avoid errors
        datetime_cols = df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns
        for col in datetime_cols:
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Extract the actual "condition" column if it exists
        condition_real = None
        if "condition" in df.columns:
            condition_real = df["condition"].tolist()
            df = df.drop(columns=["condition"])

        # Convert dataframe to list of dictionaries for flattening and filtering
        data_dicts = df.to_dict(orient="records")

        # Flatten and remove ignored columns (same as training)
        data_flat = drop_ignored_columns([flatten_dict(d) for d in data_dicts])

        # Predict with model
        preds = model.predict(data_flat)

        # Create flat DataFrame with non-dropped variables
        df_flat = pd.DataFrame(data_flat)

        # Add actual condition and prediction columns
        if condition_real is not None:
            df_flat["condition_real"] = condition_real
        df_flat["condition_pred"] = preds

        # Save flat CSV with pipe separator
        output_file = f"predictions_{uuid.uuid4().hex}.csv"
        output_path = os.path.join("technical_challenge_ml", "Static", output_file)
        df_flat.to_csv(output_path, sep="|", index=False)

        return JSONResponse({"download_link": f"/static/{output_file}"})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
