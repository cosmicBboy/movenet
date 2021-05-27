import pathlib as Path

print("TESTING ARTIFACT LOADING")

path = Path("/artifacts")
for f in path.glob("**/*"):
    print(f)
