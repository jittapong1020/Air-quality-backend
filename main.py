from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import pickle
import io
import os
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import numpy as np
import uvicorn
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Air Quality Prediction API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลดโมเดลพร้อมจัดการข้อผิดพลาด
model = None
try:
    model_path = 'rf_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✅ โหลดโมเดลสำเร็จ: {model_path}")
        logger.info(f"   Features ที่ต้องการ: {model.feature_names_in_}")
    else:
        logger.error(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
except Exception as e:
    logger.error(f"❌ ไม่สามารถโหลดโมเดลได้: {str(e)}")

# ==========================================
# ⚙️ 1. เครื่องแปลงไฟล์อัจฉริยะ
# ==========================================
def process_file_content(contents: bytes) -> pd.DataFrame:
    """
    แปลงไฟล์เป็น DataFrame รองรับทั้ง CSV และ Log ดิบ
    """
    try:
        # ลองแปลงเนื้อหาไฟล์เป็นข้อความ
        text = contents.decode('utf-8', errors='ignore')
        
        # ถ้าไฟล์ว่าง
        if not text.strip():
            return pd.DataFrame()

        # เช็คว่าในไฟล์มี "] " ไหม ถ้ามีแปลว่าเป็นไฟล์ Log ดิบ
        if "] " in text:
            return process_raw_log(text)
        else:
            # ถ้าไม่มี "] " ถือว่าเป็นไฟล์ CSV ปกติ
            return pd.read_csv(io.StringIO(text))
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return pd.DataFrame()


def process_raw_log(text: str) -> pd.DataFrame:
    """
    ประมวลผลไฟล์ Log ดิบจากเซ็นเซอร์
    """
    lines = text.strip().split('\n')
    
    headers = [
        "convert time", "unix time", "time zone", "datetime", "station",
        "x1", "x2", "x3", "long", "lat",
        "rain drop", "temp", "humudity", "pressure", "wind", "direction",
        "uv", "co2", "pm1", "pm2.5", "pm4", "pm10",
        "R", "G", "B", "infra", "intend"
    ]
    
    parsed_data = []
    for line in lines:
        line = line.strip()
        if not line or "] " not in line:
            continue
        
        # แยก header กับ info
        info = line.split("] ", 1)
        if len(info) < 2:
            continue
        
        data = info[1].split(",")
        
        # แปลง unix time → datetime
        try:
            if data and data[0].strip():
                unix_time = int(float(data[0]))  # เผื่อกรณีที่มีทศนิยม
                formatted_date = datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d %H:%M:%S")
            else:
                formatted_date = ""
        except (ValueError, IndexError, TypeError) as e:
            logger.warning(f"Cannot convert time: {e}")
            formatted_date = ""
        
        # เอาเวลาที่แปลงแล้ว ไปต่อหน้าสุด
        row = [formatted_date] + data
        
        # จัดการขนาดข้อมูลให้พอดีกับ Header
        if len(row) < len(headers):
            row += [None] * (len(headers) - len(row))
        elif len(row) > len(headers):
            row = row[:len(headers)]
            
        parsed_data.append(row)
        
    if not parsed_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(parsed_data, columns=headers)
    
    # แปลงคอลัมน์ที่เป็นตัวเลข
    numeric_cols = ['temp', 'humudity', 'pressure', 'wind', 'rain drop', 'pm2.5',
                    'pm1', 'pm4', 'pm10', 'co2', 'uv']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df


# ==========================================
# 🛡️ 2. ฟังก์ชันทำความสะอาดข้อมูล
# ==========================================
def clean_sensor_data(df):
    """
    ทำความสะอาดข้อมูลเซ็นเซอร์
    """
    if df.empty:
        return df
        
    try:
        # ลบข้อมูลซ้ำ
        if 'unix time' in df.columns:
            df = df.drop_duplicates(subset=['unix time']).copy()
        else:
            df = df.drop_duplicates().copy()

        # จัดการค่าว่าง
        df = df.ffill().bfill() 

        # ลบค่า outliers ด้วย Isolation Forest
        features = ['temp', 'humudity', 'pressure', 'pm2.5']
        existing_features = [f for f in features if f in df.columns]
        
        if len(existing_features) >= 3:  # ต้องการอย่างน้อย 3 features
            iso = IsolationForest(contamination=0.01, random_state=42, n_estimators=100)
            
            # กรองเอาเฉพาะข้อมูลที่ไม่เป็นค่าว่าง
            df_valid = df.dropna(subset=existing_features)
            if len(df_valid) > 10:  # ต้องการข้อมูลอย่างน้อย 10 แถว
                outliers = iso.fit_predict(df_valid[existing_features])
                df = df_valid[outliers == 1].copy()

        return df
        
    except Exception as e:
        logger.error(f"Error in clean_sensor_data: {str(e)}")
        return df


# ==========================================
# 📊 3. ฟังก์ชันเตรียมข้อมูลสำหรับทำนาย
# ==========================================
def prepare_data_for_prediction(df, freq='D'):
    """
    เตรียมข้อมูลสำหรับทำนายตามความถี่ที่กำหนด
    """
    if df.empty:
        return df
        
    try:
        # จัดเรียงตามเวลา
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')
            
            # Resample ตามความถี่ที่กำหนด
            if freq == 'H':
                df_resampled = df.set_index('datetime').resample('H').mean(numeric_only=True).reset_index()
            elif freq == '6H':
                df_resampled = df.set_index('datetime').resample('6H').mean(numeric_only=True).reset_index()
            elif freq == '12H':
                df_resampled = df.set_index('datetime').resample('12H').mean(numeric_only=True).reset_index()
            else:  # 'D' หรือค่าอื่นๆ
                df_resampled = df.set_index('datetime').resample('D').mean(numeric_only=True).reset_index()
        else:
            df_resampled = df.copy()
            
        return df_resampled
        
    except Exception as e:
        logger.error(f"Error in prepare_data_for_prediction: {str(e)}")
        return df


# ==========================================
# 🔮 4. ฟังก์ชันทำนายแบบ Recursive
# ==========================================
def recursive_predict(df, days, freq='D'):
    """
    ทำนายแบบ recursive ตามความถี่ที่กำหนด
    """
    if model is None:
        raise ValueError("Model not loaded")
        
    future_predictions = []
    last_data = df.copy()
    expected_features = model.feature_names_in_
    
    # กำหนดจำนวน iteration ตามความถี่
    if freq == 'H':
        n_iterations = days * 24
    elif freq == '6H':
        n_iterations = days * 4
    elif freq == '12H':
        n_iterations = days * 2
    else:  # 'D'
        n_iterations = days
    
    for i in range(n_iterations):
        try:
            last_row = last_data.iloc[-1:].copy()
            new_row = {}
            
            # สร้าง features สำหรับทำนาย
            current_pm = last_row["pm2.5"].values[0] if "pm2.5" in last_row.columns else 25.0
            
            new_row["lag1"] = current_pm
            new_row["lag2"] = last_row["lag1"].values[0] if "lag1" in last_row.columns else current_pm
            new_row["lag3"] = last_row["lag2"].values[0] if "lag2" in last_row.columns else current_pm
            new_row["lag7"] = last_data["pm2.5"].iloc[-7] if len(last_data) >= 7 else current_pm
            new_row["lag14"] = last_data["pm2.5"].iloc[-14] if len(last_data) >= 14 else current_pm
            
            # เพิ่ม features อื่นๆ ที่ model ต้องการ
            for col in expected_features:
                if col not in new_row:
                    if col in df.columns:
                        new_row[col] = df[col].mean() if not pd.isna(df[col].mean()) else 25.0
                    else:
                        new_row[col] = 25.0
                        
            # สร้าง DataFrame สำหรับทำนาย
            new_X = pd.DataFrame([new_row])
            
            # เรียงคอลัมน์ตามที่โมเดลต้องการ
            new_X = new_X[expected_features]
            
            # ทำนาย
            pred = model.predict(new_X)[0]
            pred = max(0, pred)  # ค่าฝุ่นต้องไม่ติดลบ
            
            # สร้าง timestamp
            if 'datetime' in last_data.columns and not pd.isna(last_data['datetime'].iloc[-1]):
                if freq == 'H':
                    timestamp = last_data['datetime'].iloc[-1] + timedelta(hours=1)
                elif freq == '6H':
                    timestamp = last_data['datetime'].iloc[-1] + timedelta(hours=6)
                elif freq == '12H':
                    timestamp = last_data['datetime'].iloc[-1] + timedelta(hours=12)
                else:
                    timestamp = last_data['datetime'].iloc[-1] + timedelta(days=1)
                timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp_str = f"period_{len(future_predictions) + 1}"
                
            future_predictions.append({
                'value': float(round(pred, 2)),
                'timestamp': timestamp_str
            })
            
            # เตรียมข้อมูลสำหรับรอบถัดไป
            new_row["pm2.5"] = pred
            if 'datetime' in last_data.columns:
                new_row['datetime'] = timestamp if 'timestamp' in locals() else None
                
            new_row_df = pd.DataFrame([new_row])
            last_data = pd.concat([last_data, new_row_df], ignore_index=True)
            
        except Exception as e:
            logger.error(f"Error in prediction iteration {i}: {str(e)}")
            # ถ้าผิดพลาด ให้ใช้ค่าเฉลี่ย
            future_predictions.append({
                'value': 25.0,
                'timestamp': f"period_{len(future_predictions) + 1}"
            })
        
    return future_predictions


# ==========================================
# 🎯 5. Main Prediction Endpoint
# ==========================================
@app.post("/predict")
async def predict_air_quality(
    files: List[UploadFile] = File(...),
    days: int = Form(7),
    freq: str = Form('D')
):
    """
    ทำนายค่าฝุ่น PM2.5
    - files: ไฟล์ CSV หรือไฟล์ Log ดิบ (รองรับหลายไฟล์)
    - days: จำนวนวันที่ต้องการทำนาย (1-30)
    - freq: ความถี่ ('H', '6H', '12H', 'D')
    """
    try:
        # ตรวจสอบโมเดล
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded. Please check rf_model.pkl")
            
        # ตรวจสอบจำนวนวัน
        if days < 1 or days > 30:
            raise HTTPException(status_code=400, detail="days must be between 1 and 30")
            
        # ตรวจสอบความถี่
        valid_freqs = ['H', '6H', '12H', 'D']
        if freq not in valid_freqs:
            raise HTTPException(status_code=400, detail=f"freq must be one of {valid_freqs}")
        
        logger.info(f"Received request: days={days}, freq={freq}, files={len(files)}")
        
        all_dfs = []
        for file in files:
            try:
                contents = await file.read()
                logger.info(f"Processing file: {file.filename}, size={len(contents)} bytes")
                
                df_temp = process_file_content(contents)
                
                if not df_temp.empty:
                    all_dfs.append(df_temp)
                    logger.info(f"File {file.filename}: {len(df_temp)} rows")
                else:
                    logger.warning(f"File {file.filename} is empty or invalid")
                    
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue
                
        if not all_dfs:
            raise HTTPException(status_code=400, detail="ไม่สามารถดึงข้อมูลจากไฟล์ได้ โปรดตรวจสอบรูปแบบไฟล์")
            
        # รวมข้อมูลทั้งหมด
        df_raw = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total raw data: {len(df_raw)} rows")
        
        # ทำความสะอาดข้อมูล
        df_clean = clean_sensor_data(df_raw)
        logger.info(f"After cleaning: {len(df_clean)} rows")
        
        # เตรียมข้อมูลตามความถี่
        df_prepared = prepare_data_for_prediction(df_clean, freq)
        logger.info(f"After resampling ({freq}): {len(df_prepared)} rows")
        
        # ตรวจสอบว่ามีข้อมูลเพียงพอ
        if len(df_prepared) < 14:
            raise HTTPException(status_code=400, detail=f"ข้อมูลไม่เพียงพอ ต้องการอย่างน้อย 14 จุดเวลา (มี {len(df_prepared)} จุด)")
        
        # Rename columns
        if 'humudity' in df_prepared.columns:
            df_prepared = df_prepared.rename(columns={'humudity': 'humidity'})
            
        # ตรวจสอบและจัดการคอลัมน์ pm2.5
        if 'pm2.5' not in df_prepared.columns:
            df_prepared['pm2.5'] = 25.0
            
        # สร้าง lag features
        df_prepared["lag1"] = df_prepared["pm2.5"].shift(1)
        df_prepared["lag2"] = df_prepared["pm2.5"].shift(2)
        df_prepared["lag3"] = df_prepared["pm2.5"].shift(3)
        df_prepared["lag7"] = df_prepared["pm2.5"].shift(7)
        df_prepared["lag14"] = df_prepared["pm2.5"].shift(14)
        
        # จัดการค่าว่าง
        df_prepared = df_prepared.fillna(df_prepared.mean(numeric_only=True)).fillna(25.0)
        
        logger.info("Starting recursive prediction...")
        
        # ทำนาย
        predictions = recursive_predict(df_prepared, days, freq)
        
        logger.info(f"Prediction complete: {len(predictions)} points")
        
        # จัดรูปแบบ response
        freq_names = {
            'H': 'รายชั่วโมง',
            '6H': 'ทุก 6 ชั่วโมง',
            '12H': 'ทุก 12 ชั่วโมง',
            'D': 'รายวัน'
        }
        
        time_units = {
            'H': 'ชั่วโมง',
            '6H': '6 ชั่วโมง',
            '12H': '12 ชั่วโมง',
            'D': 'วัน'
        }
        
        return {
            "success": True,
            "message": f"ทำนายค่าฝุ่น PM2.5 {freq_names[freq]} {days} วัน ({len(predictions)} จุดเวลา) ข้างหน้า",
            "frequency": freq,
            "days": days,
            "total_predictions": len(predictions),
            "predictions": [p['value'] for p in predictions],
            "timestamps": [p['timestamp'] for p in predictions],
            "time_unit": time_units[freq]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดที่ Server: {str(e)}")


# ==========================================
# 📝 6. Health Check Endpoint
# ==========================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_features": list(model.feature_names_in_) if model is not None else [],
        "supported_frequencies": ["H", "6H", "12H", "D"],
        "message": "API พร้อมทำงาน"
    }


# ==========================================
# ℹ️ 7. Info Endpoint
# ==========================================
@app.get("/info")
async def get_info():
    return {
        "name": "Air Quality Prediction API",
        "version": "2.0",
        "description": "API สำหรับทำนายค่าฝุ่น PM2.5 จากข้อมูลเซ็นเซอร์",
        "model_status": "loaded" if model is not None else "not loaded",
        "features": [
            "รองรับไฟล์ CSV และไฟล์ Log ดิบ",
            "ทำความสะอาดข้อมูลอัตโนมัติ",
            "ทำนายแบบ recursive",
            f"รองรับ {len(model.feature_names_in_) if model else 0} features" if model else "รอโหลดโมเดล"
        ],
        "input_format": {
            "files": "ไฟล์ CSV หรือไฟล์ Log ดิบ (รองรับหลายไฟล์)",
            "days": "จำนวนวันที่ต้องการทำนาย (1-30)",
            "freq": "ความถี่: 'H', '6H', '12H', 'D'"
        }
    }


# ==========================================
# 🚀 8. รันแอพพลิเคชัน
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )