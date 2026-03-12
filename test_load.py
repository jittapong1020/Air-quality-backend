import pickle
print("กำลังลองโหลดโมเดล...")
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("โหลดสำเร็จ! คุณสมบัติโมเดล:", model.feature_names_in_)