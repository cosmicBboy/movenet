from pathlib import Path

print("TESTING ARTIFACT LOADING")

path = Path("/artifacts")
for f in path.glob("**/*"):
    print(f)
