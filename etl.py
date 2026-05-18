import pandas as pd
import boto3
from sqlalchemy import create_engine

# CONFIG
BUCKET_NAME = "your-bucket-name"

# SQL connection (update this)
engine = create_engine("mysql+pymysql://user:password@host/stock_db")

s3 = boto3.client(
    's3',
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    region_name="ap-south-1"
)

# Get all files
response = s3.list_objects_v2(Bucket=BUCKET_NAME)

if "Contents" not in response:
    print("No files found in S3")
    exit()

files = sorted([obj['Key'] for obj in response['Contents']])

# Get latest file
latest_file = files[-1]

print(f"Processing: {latest_file}")

# Download file
s3.download_file(BUCKET_NAME, latest_file, "temp.csv")

# Load data
df = pd.read_csv("temp.csv")

# Insert into SQL
df.to_sql("stocks", con=engine, if_exists="append", index=False)

print("Data inserted into SQL ✅")