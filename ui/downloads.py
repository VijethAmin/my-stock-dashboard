from datetime import datetime
from io import BytesIO

import pandas as pd
import streamlit as st

from services.s3 import make_export_safe_df, upload_df_to_s3


def widget_key(*parts) -> str:
    raw = "_".join(str(part) for part in parts if part not in (None, ""))
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw)


def s3_upload_section(
    df: pd.DataFrame,
    ticker: str,
    label: str,
    s3_cfg: dict | None,
):
    if s3_cfg is None:
        st.caption(
            "☁️ S3 upload unavailable — add `[aws]` credentials to `.streamlit/secrets.toml`"
        )
        return

    col_fmt, col_btn = st.columns([1, 3])
    with col_fmt:
        fmt = st.selectbox(
            "Format", ["csv", "json"],
            key=f"s3_fmt_{ticker}_{label}",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button(f"☁️ Upload to S3 ({fmt.upper()})",
                     key=f"s3_btn_{ticker}_{label}",
                     width="stretch"):
            with st.spinner(f"Uploading {label}.{fmt} → S3…"):
                ok, msg = upload_df_to_s3(df, ticker, label, fmt, s3_cfg)
            if ok:
                st.markdown(
                    f'<div class="s3-success">✅ Uploaded successfully!<br>'
                    f'<code>{msg}</code></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="s3-error">❌ Upload failed: {msg}</div>',
                    unsafe_allow_html=True,
                )


def create_download_buttons(
    df: pd.DataFrame,
    filename_prefix: str,
    ticker: str = "DATA",
    label: str = "",
    s3_cfg: dict | None = None,
):
    label = label or filename_prefix
    stamp = datetime.now().strftime("%Y%m%d")
    key_base = widget_key(filename_prefix, ticker, label)

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button(
            "📥 Download CSV",
            csv,
            f"{filename_prefix}_{stamp}.csv",
            mime="text/csv",
            width="stretch",
            key=f"csv_{key_base}",
        )

    with col2:
        buf = BytesIO()
        df_exp = make_export_safe_df(df)
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df_exp.to_excel(writer, index=True, sheet_name="Data")
        st.download_button(
            "📥 Download Excel", buf.getvalue(),
            f"{filename_prefix}_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
            key=f"excel_{key_base}",
        )

    with col3:
        if s3_cfg is None:
            st.button("☁️ S3 (not configured)", disabled=True,
                      width="stretch",
                      key=f"s3_disabled_{key_base}")
        else:
            if st.button("☁️ Upload to S3",
                         key=f"s3_{key_base}",
                         width="stretch"):
                with st.spinner("Uploading…"):
                    ok, msg = upload_df_to_s3(df, ticker, label, "csv", s3_cfg)
                if ok:
                    st.success(f"✅ `{msg}`")
                else:
                    st.error(f"❌ {msg}")

