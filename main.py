from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import pickle
import io
from sklearn.ensemble import IsolationForest
from datetime import datetime # 🚨 นำเข้าไลบรารีนี้เพิ่ม เพื่อใช้แปลงเวลาแบบ sensor.py

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ==========================================
# ⚙️ 1. เครื่องแปลงไฟล์อัจฉริยะ (ผสมจาก sensor.py)
# ==========================================
def process_file_content(contents: bytes) -> pd.DataFrame:
    # ลองแปลงเนื้อหาไฟล์เป็นข้อความ
    text = contents.decode('utf-8', errors='ignore')

    # เช็คว่าในไฟล์มี "] " ไหม ถ้ามีแปลว่าเป็นไฟล์ Log ดิบ (ต้นแบบ)
    if "] " in text:
        lines = text.strip().split('\n')
        
        # เตรียม Header (เปลี่ยน x เป็น x1, x2, x3 เพื่อไม่ให้คอลัมน์ซ้ำกันจน pandas งง)
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
                unix_time = int(data[0])
                formatted_date = datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, IndexError):
                formatted_date = ""
            
            # เอาเวลาที่แปลงแล้ว ไปต่อหน้าสุด
            row = [formatted_date] + data
            
            # จัดการขนาดข้อมูลให้พอดีกับ Header
            if len(row) < len(headers):
                row += [None] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[:len(headers)]
                
            parsed_data.append(row)
            
        df = pd.DataFrame(parsed_data, columns=headers)
        
        # 🚨 จุดสำคัญ: บังคับแปลงคอลัมน์ที่เป็นตัวอักษรให้กลายเป็น "ตัวเลข" ก่อนส่งให้ AI
        numeric_cols = ['temp', 'humudity', 'pressure', 'wind', 'rain drop', 'pm2.5']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
        
    else:
        # ถ้าไม่มี "] " ถือว่าเป็นไฟล์ CSV ปกติ
        return pd.read_csv(io.StringIO(text))


# ==========================================
# 🛡️ 2. ฟังก์ชันทำความสะอาดข้อมูล (Data Cleaning)
# ==========================================
def clean_sensor_data(df):
    if 'unix time' in df.columns:
        df = df.drop_duplicates(subset=['unix time']).copy()
    else:
        df = df.drop_duplicates().copy()

    df = df.ffill().bfill() 

    features = ['temp', 'humudity', 'pressure', 'pm2.5']
    existing_features = [f for f in features if f in df.columns]
    
    if len(existing_features) > 0:
        iso = IsolationForest(contamination=0.01, random_state=42)
        
        # กรองเอาเฉพาะข้อมูลที่ไม่เป็นค่าว่าง (NaN) ไปเช็ค Outlier
        df_valid = df.dropna(subset=existing_features)
        if not df_valid.empty:
            outliers = iso.fit_predict(df_valid[existing_features])
            df = df_valid[outliers == 1].copy()

    return df
# ==========================================


@app.post("/predict")
async def predict_air_quality(
    files: List[UploadFile] = File(...),
    days: int = Form(7)
):
    try:
        all_dfs = []
        for file in files:
            contents = await file.read()
            
            # 🚨 1. ส่งไฟล์เข้าเครื่องแปลงอัจฉริยะ (รับจบทั้ง Log ดิบ และ CSV)
            df_temp = process_file_content(contents)
            
            if not df_temp.empty:
                all_dfs.append(df_temp)
                
        if not all_dfs:
            return {"error": "ไม่สามารถดึงข้อมูลจากไฟล์ได้ โปรดตรวจสอบรูปแบบไฟล์"}
            
        df_raw = pd.concat(all_dfs, ignore_index=True)
        
        # 🚨 2. ส่งข้อมูลดิบเข้าเครื่องซักผ้าทำความสะอาด
        df_clean = clean_sensor_data(df_raw)
        
        # 3. จัดเรียงและทำค่าเฉลี่ยรายวัน
        if 'datetime' in df_clean.columns:
            df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
            df_clean = df_clean.sort_values('datetime')
            df_daily = df_clean.set_index('datetime').resample('D').mean(numeric_only=True).reset_index()
        else:
            df_daily = df_clean.copy()
            
        if 'humudity' in df_daily.columns:
            df_daily = df_daily.rename(columns={'humudity': 'humidity'})
            
        if 'pm2.5' not in df_daily.columns:
            df_daily['pm2.5'] = 25.0
            
        df_daily["lag1"] = df_daily["pm2.5"].shift(1)
        df_daily["lag2"] = df_daily["pm2.5"].shift(2)
        df_daily["lag3"] = df_daily["pm2.5"].shift(3)
        df_daily["lag7"] = df_daily["pm2.5"].shift(7)
        df_daily["lag14"] = df_daily["pm2.5"].shift(14)
        
        df_daily = df_daily.fillna(df_daily.mean(numeric_only=True)).fillna(25.0)

        # 4. วนลูปทำนายอนาคตทีละวัน
        future_predictions = []
        last_data = df_daily.copy()
        expected_features = model.feature_names_in_
        
        for i in range(days):
            last_row = last_data.iloc[-1:].copy()
            new_row = {}
            
            new_row["lag1"] = last_row["pm2.5"].values[0]
            new_row["lag2"] = last_row["lag1"].values[0] if "lag1" in last_row.columns else 25.0
            new_row["lag3"] = last_row["lag2"].values[0] if "lag2" in last_row.columns else 25.0
            new_row["lag7"] = last_data["pm2.5"].iloc[-7] if len(last_data) >= 7 else 25.0
            new_row["lag14"] = last_data["pm2.5"].iloc[-14] if len(last_data) >= 14 else 25.0
            
            for col in expected_features:
                if col not in new_row:
                    if col in df_daily.columns:
                        new_row[col] = df_daily[col].mean()
                    else:
                        new_row[col] = 25.0
                        
            new_X = pd.DataFrame([new_row])[expected_features]
            pred = model.predict(new_X)[0]
            future_predictions.append(pred)
            
            new_row["pm2.5"] = pred
            last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)
            
        return {"predictions": future_predictions}
    
    except Exception as e:
        return {"error": f"เกิดข้อผิดพลาดที่ Server: {str(e)}"}