import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


def download_data():
    api = KaggleApi()
    api.authenticate()

    competition = "playground-series-s4e7"
    api.competition_download_files(competition, path="data/raw/")

    with zipfile.ZipFile("data/raw/playground-series-s4e7.zip", "r") as zip_ref:
        zip_ref.extractall("data/raw/")

    print("Data downloaded and extracted.")


if __name__ == "__main__":
    download_data()
