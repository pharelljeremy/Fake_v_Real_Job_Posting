import argparse
from src.config import ensure_dirs, logger
ensure_dirs()

def main(mode: str = "all"):
    logger.info("Running pipeline mode=%s", mode)
    # Step order: ingest -> clean -> analyze -> visualise -> train
    # We call modules directly to keep single-process and share filesystem
    from src.ingest import load_raw
    from src.clean import clean
    from src.analyze import run_analysis
    from src.visualise import run_visuals
    from src.train import train_batch, train_incremental

    # Ingest & clean
    df_raw = load_raw()
    df_clean = clean(df_raw)

    # Analysis & visuals
    run_analysis()
    run_visuals()

    # Train
    if mode in ("batch", "all"):
        train_batch()
    if mode in ("incremental", "all"):
        train_incremental()

    logger.info("Pipeline finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "incremental", "all"], default="all")
    args = parser.parse_args()
    main(mode=args.mode)

