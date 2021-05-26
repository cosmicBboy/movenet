from pathlib import Path

print("gridai test script")

data_path = Path("/opt/datastore")

for f in data_path.glob("**/*"):
    print(f)
