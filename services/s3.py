import os
from datetime import datetime
from io import BytesIO

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


def get_s3_config():
    try:
        aws_secrets = st.secrets.get("aws", {})
    except Exception:
        aws_secrets = {}

    bucket = (
        aws_secrets.get("bucket")
        or aws_secrets.get("bucket_name")
        or os.getenv("AWS_S3_BUCKET")
    )
    region = aws_secrets.get("region") or os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

    if not bucket:
        return None

    cfg = {"bucket": bucket, "region": region}
    if aws_secrets.get("access_key_id") and aws_secrets.get("secret_access_key"):
        cfg["access_key_id"] = aws_secrets["access_key_id"]
        cfg["secret_access_key"] = aws_secrets["secret_access_key"]
    return cfg


@st.cache_resource(show_spinner=False)
def build_s3_client(cfg=None):
    kwargs = {}
    if cfg:
        kwargs["region_name"] = cfg.get("region")
        if cfg.get("access_key_id") and cfg.get("secret_access_key"):
            kwargs["aws_access_key_id"] = cfg["access_key_id"]
            kwargs["aws_secret_access_key"] = cfg["secret_access_key"]
    return boto3.client("s3", **kwargs)


def make_export_safe_df(df: pd.DataFrame, reset_index: bool = False) -> pd.DataFrame:
    df_exp = df.copy()
    if reset_index:
        df_exp = df_exp.reset_index()

    if isinstance(df_exp.index, pd.DatetimeIndex) and df_exp.index.tz is not None:
        df_exp.index = df_exp.index.tz_localize(None)

    for col in df_exp.select_dtypes(include=["datetimetz"]).columns:
        df_exp[col] = df_exp[col].dt.tz_localize(None)

    return df_exp


def s3_key(ticker: str, label: str, fmt: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"stocks/{ticker}/{date_str}/{label}.{fmt}"


def upload_df_to_s3(
    df: pd.DataFrame,
    ticker: str,
    label: str,
    fmt: str = "csv",
    cfg: dict | None = None,
) -> tuple[bool, str]:
    if cfg is None:
        return False, "S3 config missing"
    try:
        buf = BytesIO()
        df_exp = make_export_safe_df(df, reset_index=True)

        if fmt == "csv":
            buf.write(df_exp.to_csv(index=False).encode("utf-8"))
            content_type = "text/csv"
        else:
            buf.write(
                df_exp.where(pd.notna(df_exp), None)
                .to_json(orient="records", date_format="iso", indent=2)
                .encode("utf-8")
            )
            content_type = "application/json"

        buf.seek(0)
        s3 = build_s3_client(cfg)
        key = s3_key(ticker, label, fmt)
        s3.put_object(
            Bucket=cfg["bucket"],
            Key=key,
            Body=buf.getvalue(),
            ContentType=content_type,
        )
        return True, f"s3://{cfg['bucket']}/{key}"

    except NoCredentialsError:
        return False, "AWS credentials are invalid or expired."
    except ClientError as e:
        code = e.response["Error"]["Code"]
        msg = e.response["Error"]["Message"]
        return False, f"S3 error [{code}]: {msg}"
    except Exception as e:
        return False, f"Unexpected error: {e}"
