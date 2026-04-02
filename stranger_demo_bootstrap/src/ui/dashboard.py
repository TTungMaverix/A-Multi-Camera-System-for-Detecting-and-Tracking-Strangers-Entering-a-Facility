from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Stranger Demo Dashboard", layout="wide")
st.title("Stranger Demo Dashboard")
st.caption("Dashboard nay la lop trinh dien tuy chon. Output chinh van la files va logs.")

events_path = Path("data/outputs/events/events.csv")
profiles_path = Path("data/db/stranger_profiles.csv")

left, right = st.columns(2)

with left:
    st.subheader("Event Table")
    if events_path.exists():
        events_df = pd.read_csv(events_path)
        st.dataframe(events_df, use_container_width=True)
    else:
        st.info("Chua co file data/outputs/events/events.csv")

with right:
    st.subheader("Stranger Profiles")
    if profiles_path.exists():
        profiles_df = pd.read_csv(profiles_path)
        st.dataframe(profiles_df, use_container_width=True)
    else:
        st.info("Chua co file data/db/stranger_profiles.csv")

st.subheader("Presentation Direction")
st.markdown(
    """
    - Co the show demo theo dang web hoac quad-view 4 camera.
    - Overlay de xuat: `camera_id`, `local_track_id`, `global_id`, `identity_type`, `direction`.
    - Layer nay dung de thuyet trinh; log files moi la output chinh cua de tai.
    """
)
