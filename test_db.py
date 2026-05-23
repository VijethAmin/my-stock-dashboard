import streamlit as st
import pandas as pd

try:
    conn = st.connection("mssql", type="sql")
    df = conn.query("SELECT 1 AS test")
    st.write("Connection successful!")
    st.write(df)
except Exception as e:
    st.error(f"Connection failed: {str(e)}")