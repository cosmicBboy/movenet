from pathlib import Path

print("gridai test script")

data_path = Path("/kinetics_dataset")
print(data_path)

for f in data_path.glob("**/*"):
    print(f)

print("done")
