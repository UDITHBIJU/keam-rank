import joblib

with open("college_model.pkl", "rb") as f:
    loaded_objects = joblib.load(f)

print(f"Loaded {len(loaded_objects)} objects.")
for idx, obj in enumerate(loaded_objects):
    print(f"Object {idx+1}: Type -> {type(obj)}")
