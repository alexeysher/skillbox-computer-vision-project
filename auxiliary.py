import streamlit as st
from pathlib import Path
from google.cloud import storage
from google.cloud import aiplatform
from google.oauth2 import service_account
import re
import plotly.express as px


def css_styling():
    """
    Styles UI.
    """
    st.html(f"""
    <style>
        MainMenu {{visibility: hidden;}}
        # header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        # .st-emotion-cache-1104ytp {{
        # }}
        # .st-emotion-cache-1104ytp h4 {{
        #     padding-bottom: 0rem;
        # }}
        .st-emotion-cache-hk58ed {{
            font-size: 0.85rem;
        }}
    </style>
    """)


def rgb_to_rgba(color: str, opacity: float = 1.0):
    rgb = re.findall(r'\d+', color)
    rgba = rgb + [str(opacity)]
    return 'rgba(' + ', '.join(rgba)  + ')'


DEFAULT_COLORS = [
    rgb_to_rgba(rgb, 0.75) for rgb in px.colors.DEFAULT_PLOTLY_COLORS
]


class GoogleCloud:

    def __init__(self):
        credentials = service_account.Credentials.from_service_account_info(st.secrets['gc-service-account'])
        storage_client = storage.Client(project=credentials.project_id, credentials=credentials)
        self.bucket = storage_client.bucket(st.secrets['gc-storage']['bucket_id'])
        self.endpoints =[
            aiplatform.Endpoint(
                project=credentials.project_id,
                location='us-central1',
                endpoint_name=name,
                credentials=credentials
            )
            for name in st.secrets['gc-aiplatform']['endpoint_ids']
        ]
        self.url = f'https://storage.cloud.google.com/{self.bucket.name}'

    def _create_blob(self, file_path: str | Path):
        """
        Creates blob for file operations with file on GC.
        """
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        blob = self.bucket.blob(file_path)
        return blob

    def download_file(self, file_path: str | Path, downloaded_file_path: str | Path) -> bool:
        """
        Downloads file from GC.
        """
        if isinstance(downloaded_file_path, Path):
            downloaded_file_path = downloaded_file_path.as_posix()
        blob = self._create_blob(file_path)
        try:
            blob.download_to_filename(downloaded_file_path)
        except:
            return False
        return True

    def open_file(self, file_path: str | Path, mode='r'):
        """
        Create a file handler for file-like I/O from GC.
        """
        blob = self._create_blob(file_path)
        return blob.open(mode=mode)

    def upload_file(self, file_path: str | Path, uploading_file_path: str | Path) -> bool:
        """
        Uploads file to GC.
        """
        if isinstance(uploading_file_path, Path):
            uploading_file_path = uploading_file_path.as_posix()
        blob = self._create_blob(file_path)
        try:
            blob.upload_from_filename(uploading_file_path)
        except:
            return False
        return True

    def delete_file(self, file_path: str):
        blob = self._create_blob(file_path)
        if blob.exists():
            blob.delete()

    def get_blobs(self, pattern: str | None = None):
        blobs = self.bucket.list_blobs(match_glob=pattern)
        return blobs

    def get_blob_urls(self, pattern: str | None = None):
        urls = [
            self.url + '/' + blob.name for blob in self.get_blobs(pattern)
        ]
        return urls

    def delete_blobs(self, pattern: str | None = None):
        for blob in self.get_blobs(pattern):
            blob.delete()


