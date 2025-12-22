import streamlit as st
import time
import random
from pathlib import Path
import os
from typing import List
from longtail_system import longtail_system
# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = Path("bdd100") 
IMAGES_PER_ROW = 5
NUM_INITIAL_IMAGES = 15
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -----------------------------
# Utility functions
# -----------------------------
def load_all_images(dataset_dir: Path) -> List[Path]:
    return [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def sample_random_images(images: List[Path], k: int) -> List[Path]:
    if len(images) <= k:
        return images
    return random.sample(images, k)


def search_images(query: str, images: List[Path]) -> List[Path]:
    results = longtail_system(query,DATASET_DIR)
    results = [os.path.join(DATASET_DIR,nome) for nome in results]
        
    return results


def display_image_grid(image_paths: List[Path], images_per_row: int):
    rows = [
        image_paths[i:i + images_per_row]
        for i in range(0, len(image_paths), images_per_row)
    ]

    for row in rows:
        cols = st.columns(images_per_row)
        for col, img_path in zip(cols, row):
            with col:
                st.image(str(img_path), use_container_width=True)


# -----------------------------
# App state initialization
# -----------------------------
if "all_images" not in st.session_state:
    st.session_state.all_images = load_all_images(DATASET_DIR)

if "current_images" not in st.session_state:
    st.session_state.current_images = sample_random_images(
        st.session_state.all_images,
        NUM_INITIAL_IMAGES
    )

if "last_search_time" not in st.session_state:
    st.session_state.last_search_time = None


# -----------------------------
# UI
# -----------------------------
st.set_page_config(layout="wide")
#st.title("demo")

query = st.text_input(
    label="",
    placeholder="car overtaking at night"
)

send_clicked = st.button("Send")

# -----------------------------
# Search logic
# -----------------------------
if send_clicked and query.strip():
    with st.spinner("Searching database..."):
        start_time = time.perf_counter()
        results = search_images(query, st.session_state.all_images)
        elapsed = time.perf_counter() - start_time

    st.session_state.current_images = results
    st.session_state.last_search_time = elapsed

# -----------------------------
# Feedback
# -----------------------------
if st.session_state.last_search_time is not None:
    st.caption(f"Search completed in {st.session_state.last_search_time:.3f} seconds")

# -----------------------------
# Image display
# -----------------------------
display_image_grid(
    st.session_state.current_images,
    IMAGES_PER_ROW
)
