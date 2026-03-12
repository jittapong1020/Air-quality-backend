# PM2.5 Prediction API (FastAPI) 🚀

ระบบ API สำหรับพยากรณ์ค่าฝุ่น PM2.5 ล่วงหน้า โดยใช้โมเดล Machine Learning (Random Forest) และประมวลผลข้อมูลจากไฟล์ Log หรือ CSV ของเซนเซอร์ตรวจวัดคุณภาพอากาศ

## 🛠️ Tech Stack
* **Framework:** FastAPI (Python)
* **ML Library:** Scikit-learn, Pandas, NumPy
* **Model:** Random Forest Regressor (`rf_model.pkl`)
* **Server:** Uvicorn (ASGI)
* **Deployment:** Render (Dockerized)

## 📁 โครงสร้างโฟลเดอร์
```text
.
├── main.py              # ไฟล์หลักของ API และ logic การประมวลผลไฟล์
├── rf_model.pkl         # ไฟล์โมเดลที่เทรนสำเร็จแล้ว
├── requirements.txt     # รายการ Library ที่ต้องใช้
├── Dockerfile           # สำหรับการ Deploy บน Render/Cloud
└── README.md            # คู่มือการใช้งาน