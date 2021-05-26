from pathlib import Path

print("gridai test script")

data_path = Path("/opt/datastore")
print(data_path)

for f in data_path.glob("**/*"):
    print(f)

print("done")
