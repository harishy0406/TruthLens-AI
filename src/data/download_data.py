import os
import subprocess
import kagglehub
from kagglehub import KaggleDatasetAdapter


# DATASET 1

# Set the path to the file you'd like to load
file_path = "data/external"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "clmentbisaillon/fake-and-real-news-dataset",
  file_path,
)

print("First 5 records:", df.head())


# DATASET 2
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "data/external"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "atharvasoundankar/gen-ai-misinformation-detection-datase-20242025",
  file_path,
)

print("First 5 records:", df.head())