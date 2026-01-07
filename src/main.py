from src.ingest import load_raw
from src.clean import clean
from src.analyze import run_analysis
from src.visualise import run_visuals

def main():
    print("Ingesting raw data...")
    df = load_raw()
    print("Cleaning data...")
    dfc = clean(df)
    print("Running analysis...")
    run_analysis()
    print("Generating visuals...")
    run_visuals()
    print("Done.")

if __name__ == "__main__":
    main()

