from pathlib import Path

def main():
    for p in ["data/weather","data/markets","data/sports","app"]:
        Path(p).mkdir(parents=True, exist_ok=True)
    print("Folders ready.")

if __name__ == "__main__":
    main()
