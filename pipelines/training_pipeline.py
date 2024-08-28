import subprocess


def run_pipeline():
    subprocess.run(["python", "data/fetch_data.py"])

    subprocess.run(["python", "data/clean_data.py"])

    subprocess.run(["python", "models/train_model.py"])

    subprocess.run(["python", "models/evaluate_model.py"])


if __name__ == "__main__":
    run_pipeline()
