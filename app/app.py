import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import io

st.set_page_config(
    page_title="Eigenfaces",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
    /* Base */
    html, body, [class*="css"] {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0d1117;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] * {
        color: #e6edf3;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-family: 'Courier New', monospace !important;
        color: #e6edf3 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-family: 'Courier New', monospace;
        color: #8b949e;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #00e5ff;
        border-bottom-color: #00e5ff;
    }

    /* Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* Placeholder */
    .upload-placeholder {
        background: #161b22;
        border: 2px dashed #30363d;
        border-radius: 8px;
        padding: 3rem;
        text-align: center;
        color: #8b949e;
        font-family: 'Courier New', monospace;
    }

    /* Slider accent */
    [data-testid="stSlider"] .st-ae {
        background: #00e5ff;
    }

    /* Math blurb code block */
    .math-blurb {
        background: #0d1117;
        border: 1px solid #30363d;
        border-left: 3px solid #00e5ff;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        color: #00e5ff;
        white-space: pre;
        line-height: 1.7;
    }

    /* Muted text */
    .muted {
        color: #8b949e;
        font-size: 0.82rem;
    }

    /* Accent text */
    .accent {
        color: #00e5ff;
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#e6edf3", family="Courier New"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    margin=dict(l=40, r=20, t=40, b=40),
)

CYAN = "#00e5ff"


@st.cache_resource
def load_eigenfaces():
    from sklearn.datasets import fetch_olivetti_faces
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data
    target = data.target
    mean_face = X.mean(axis=0)
    X_centered = X - mean_face
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = s ** 2 / (X.shape[0] - 1)
    eigenfaces = Vt
    coords = X_centered @ eigenfaces.T
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    return X, target, mean_face, eigenvalues, eigenfaces, coords, cumvar


def reconstruct(face, mean_face, eigenfaces, k):
    centered = face - mean_face
    coeffs = eigenfaces[:k] @ centered
    return np.clip(mean_face + eigenfaces[:k].T @ coeffs, 0, 1)


def face_to_img(arr_1d, n=64):
    return (np.clip(arr_1d, 0, 1).reshape(n, n) * 255).astype(np.uint8)


@st.cache_resource
def build_eigenface_grid_images(_eigenfaces):
    imgs = []
    for i in range(40):
        ef = _eigenfaces[i]
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-9)
        imgs.append(face_to_img(ef_norm))
    return imgs


X, target, mean_face, eigenvalues, eigenfaces, coords, cumvar = load_eigenfaces()
n_pixels = 64

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h2 style='font-family:Courier New; color:#00e5ff;'>⟨eigenfaces⟩</h2>",
                unsafe_allow_html=True)
    st.markdown("<p class='muted'>Spectral Theorem applied to a dataset of human faces.</p>",
                unsafe_allow_html=True)
    st.divider()

    k = st.slider("k — number of components", min_value=1, max_value=150, value=30)

    st.divider()
    st.markdown("""<div class='math-blurb'>f̂ₖ = f̄ + Σᵢ₌₁ᵏ ⟨f−f̄, φᵢ⟩ φᵢ</div>""",
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p class='muted'><a href='https://github.com' style='color:#00e5ff;'>GitHub →</a></p>",
                unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Eigenfaces", "Reconstruct", "Upload", "Explore"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Eigenfaces
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.markdown("## The basis of face space")
    st.markdown(
        "Each image below is an eigenvector of the covariance matrix, reshaped to 64×64. "
        "Together, they span the space of all faces in the dataset."
    )

    grid_imgs = build_eigenface_grid_images(eigenfaces)

    cols_per_row = 10
    for row in range(4):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            i = row * cols_per_row + col_idx
            col.image(grid_imgs[i], caption=f"φ{i+1}", use_container_width=True)

    st.divider()

    # Eigenvalue spectrum bar chart
    st.markdown("### Eigenvalue spectrum (top 100)")
    opacities = [1.0 - 0.7 * (i / 99) for i in range(100)]
    colors = [f"rgba(0,229,255,{o:.2f})" for o in opacities]

    fig_bar = go.Figure(go.Bar(
        x=list(range(1, 101)),
        y=eigenvalues[:100].tolist(),
        marker_color=colors,
        showlegend=False,
    ))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Component index i",
        yaxis_title="Eigenvalue λᵢ",
        title="Eigenvalue spectrum",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Cumulative variance
    st.markdown("### Cumulative variance explained")
    k90 = int(np.searchsorted(cumvar, 0.90)) + 1
    k95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=list(range(1, 401)),
        y=(cumvar * 100).tolist(),
        line=dict(color=CYAN, width=2),
        name="Cumulative variance",
    ))
    fig_var.add_hline(y=90, line_dash="dash", line_color="#ff6b6b",
                      annotation_text=f"90% → k={k90}",
                      annotation_font_color="#ff6b6b")
    fig_var.add_hline(y=95, line_dash="dash", line_color="#ffd93d",
                      annotation_text=f"95% → k={k95}",
                      annotation_font_color="#ffd93d")
    fig_var.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Number of components k",
        yaxis_title="Cumulative variance explained (%)",
        title="How many eigenfaces do we need?",
    )
    st.plotly_chart(fig_var, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Reconstruct
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    if "face_idx" not in st.session_state:
        st.session_state.face_idx = 42

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Choose a face")
        face_idx = st.number_input(
            "Face index (0–399)", min_value=0, max_value=399,
            value=st.session_state.face_idx, key="face_input"
        )
        st.session_state.face_idx = face_idx

        if st.button("Randomize"):
            st.session_state.face_idx = int(np.random.randint(0, 400))
            st.rerun()

        selected_face = X[st.session_state.face_idx]
        st.image(
            face_to_img(selected_face),
            caption=f"Subject #{target[st.session_state.face_idx]}",
            width=200,
        )

    with col_right:
        st.markdown(f"### Reconstruction with k={k} components")
        recon = reconstruct(selected_face, mean_face, eigenfaces, k)
        pct = cumvar[k - 1] * 100
        st.image(
            face_to_img(recon),
            caption=f"{pct:.1f}% of variance · {k} numbers",
            width=200,
        )

    st.divider()

    # Reconstruction error vs k
    st.markdown("### Reconstruction error vs. k")
    face_centered = selected_face - mean_face
    ks_range = list(range(1, 151))
    errors = []
    for ki in ks_range:
        r = reconstruct(selected_face, mean_face, eigenfaces, ki)
        errors.append(float(np.sum((selected_face - r) ** 2)))

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=ks_range, y=errors,
        line=dict(color=CYAN, width=2),
        name="Reconstruction error",
    ))
    fig_err.add_vline(
        x=k, line_dash="dash", line_color="#ffd93d",
        annotation_text=f"k={k}",
        annotation_font_color="#ffd93d",
    )
    fig_err.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="k (components)",
        yaxis_title="‖f − f̂ₖ‖²",
        title=f"Reconstruction error for face #{st.session_state.face_idx}",
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Upload
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.markdown("## Reconstruct your own face")
    st.markdown(
        "<p class='muted'>The eigenfaces were computed from a dataset of 400 images. "
        "Reconstruction quality depends on how similar your image is to that distribution.</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            img_bytes = uploaded.read()
            img_pil = Image.open(io.BytesIO(img_bytes)).convert("L").resize((64, 64))
            img_arr = np.array(img_pil, dtype=np.float64) / 255.0
            face_upload = img_arr.flatten()

            recon_upload = reconstruct(face_upload, mean_face, eigenfaces, k)
            err_upload = float(np.sum((face_upload - recon_upload) ** 2))
            pct_upload = cumvar[k - 1] * 100

            col1, col2, col3 = st.columns(3)
            col1.image(face_to_img(face_upload), caption="Your image (64×64 grayscale)", width=180)
            col2.image(face_to_img(mean_face), caption="Mean face", width=180)
            col3.image(face_to_img(recon_upload),
                       caption=f"Reconstruction k={k}\n({pct_upload:.1f}% variance)", width=180)

            st.markdown(
                f"<p class='muted'>Reconstruction error: <span class='accent'>"
                f"{err_upload:.4f}</span> &nbsp;|&nbsp; "
                f"Variance explained: <span class='accent'>{pct_upload:.1f}%</span></p>",
                unsafe_allow_html=True,
            )

            # Error vs k for uploaded image
            st.markdown("### Reconstruction error vs. k")
            ks_up = list(range(1, 151))
            errs_up = []
            for ki in ks_up:
                r = reconstruct(face_upload, mean_face, eigenfaces, ki)
                errs_up.append(float(np.sum((face_upload - r) ** 2)))

            fig_up = go.Figure()
            fig_up.add_trace(go.Scatter(
                x=ks_up, y=errs_up,
                line=dict(color=CYAN, width=2),
                name="Error",
            ))
            fig_up.add_vline(
                x=k, line_dash="dash", line_color="#ffd93d",
                annotation_text=f"k={k}",
                annotation_font_color="#ffd93d",
            )
            fig_up.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="k (components)",
                yaxis_title="‖f − f̂ₖ‖²",
                title="Reconstruction error for uploaded image",
            )
            st.plotly_chart(fig_up, use_container_width=True)

        except Exception as e:
            st.error(f"Could not process the uploaded image: {e}. "
                     "Please upload a valid JPG or PNG file.")
    else:
        st.markdown("""
        <div class='upload-placeholder'>
            drag and drop an image here<br>
            <span style='font-size:0.75rem; color:#30363d;'>JPG or PNG · any size · will be resized to 64×64</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Explore — walk along an eigenface direction
# ─────────────────────────────────────────────────────────────────────────────

with tab4:
    st.markdown("## Walk along an eigenface direction")
    st.markdown(
        "Each eigenface defines a direction in the 4096-dimensional face space. "
        "Moving a face along that direction exaggerates or suppresses the corresponding "
        "mode of variation in the dataset."
    )
    st.markdown(
        "<div class='card'>"
        "<span class='accent' style='font-family:Courier New'>f_new = clip(f + α · √λᵢ · φᵢ, 0, 1)</span><br>"
        "<span class='muted'>α is in standard deviations — so ±3 is an extreme but valid displacement. "
        "√λᵢ scales the step so that one unit = one standard deviation of natural variation along φᵢ.</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p class='muted'><b>Note on φ₁:</b> The Olivetti dataset is nearly homogeneous in race "
        "(mostly white male academics, Cambridge UK, early 1990s). "
        "φ₁ most likely captures <em>illumination</em> — the largest source of variance is "
        "lighting direction and overall brightness, not skin tone. "
        "Moving along −φ₁ will darken the face, but the precise reading is <em>shadows, not ethnicity</em>. "
        "Still a great demo: notice whether the eye whites are affected less than the skin.</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Face source ────────────────────────────────────────────────────────────
    source = st.radio("Face source", ["Dataset face", "Upload image"],
                      horizontal=True, label_visibility="collapsed")

    explore_face = None

    if source == "Dataset face":
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if "explore_idx" not in st.session_state:
                st.session_state.explore_idx = 7
            explore_idx = st.number_input(
                "Face index (0–399)", min_value=0, max_value=399,
                value=st.session_state.explore_idx, key="explore_input",
            )
            st.session_state.explore_idx = explore_idx
            if st.button("Randomize", key="explore_rand"):
                st.session_state.explore_idx = int(np.random.randint(0, 400))
                st.rerun()
        explore_face = X[st.session_state.explore_idx]

    else:
        up_explore = st.file_uploader("Upload image for exploration",
                                      type=["jpg", "jpeg", "png"],
                                      key="explore_upload")
        if up_explore is not None:
            try:
                img_pil = Image.open(io.BytesIO(up_explore.read())).convert("L").resize((64, 64))
                explore_face = np.array(img_pil, dtype=np.float64).flatten() / 255.0
            except Exception as e:
                st.error(f"Could not load image: {e}")
        else:
            st.markdown("""
            <div class='upload-placeholder'>
                upload an image above to begin
            </div>""", unsafe_allow_html=True)

    if explore_face is not None:
        st.divider()

        # ── Controls ───────────────────────────────────────────────────────────
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            ef_index = st.slider(
                "Eigenface direction φᵢ  (i =)",
                min_value=1, max_value=20, value=1,
                help="Which eigenvector direction to walk along.",
            )
        with ctrl_col2:
            alpha = st.slider(
                "α  (standard deviations)",
                min_value=-3.0, max_value=3.0, value=0.0, step=0.1,
                help="Displacement along φᵢ in units of √λᵢ. "
                     "α=0 is the original face; ±3 is a large but realistic step.",
            )

        # ── Compute ────────────────────────────────────────────────────────────
        i = ef_index - 1
        std_i = float(np.sqrt(eigenvalues[i]))
        phi_i = eigenfaces[i]
        modified_face = np.clip(explore_face + alpha * std_i * phi_i, 0, 1)

        # Normalize eigenface for display (it has negative values)
        ef_display = eigenfaces[i]
        ef_display_norm = (ef_display - ef_display.min()) / (ef_display.max() - ef_display.min() + 1e-9)

        # ── Display ────────────────────────────────────────────────────────────
        img_col1, img_col2, img_col3 = st.columns(3)

        with img_col1:
            st.markdown("<p style='text-align:center; color:#8b949e; font-family:Courier New'>original</p>",
                        unsafe_allow_html=True)
            st.image(face_to_img(explore_face), use_container_width=True)

        with img_col2:
            st.markdown(
                f"<p style='text-align:center; color:#8b949e; font-family:Courier New'>"
                f"φ<sub>{ef_index}</sub> &nbsp;(λ = {eigenvalues[i]:.2f})</p>",
                unsafe_allow_html=True,
            )
            st.image(face_to_img(ef_display_norm), use_container_width=True)
            st.markdown(
                f"<p class='muted' style='text-align:center'>"
                f"bright pixels → + direction<br>dark pixels → − direction</p>",
                unsafe_allow_html=True,
            )

        with img_col3:
            direction_label = f"+{alpha:.1f}σ" if alpha >= 0 else f"{alpha:.1f}σ"
            st.markdown(
                f"<p style='text-align:center; color:#00e5ff; font-family:Courier New'>"
                f"f + ({direction_label}) · √λ · φ<sub>{ef_index}</sub></p>",
                unsafe_allow_html=True,
            )
            st.image(face_to_img(modified_face), use_container_width=True)

        # ── Per-pixel difference chart ─────────────────────────────────────────
        st.divider()
        st.markdown("### Pixel-region breakdown of the displacement")
        st.markdown(
            "<p class='muted'>The bar chart below shows the mean absolute change per "
            "image region. If φᵢ carries more weight in the skin regions than in "
            "the eyes, the eye whites should be less affected by the traversal.</p>",
            unsafe_allow_html=True,
        )

        diff = np.abs(modified_face - explore_face).reshape(64, 64)

        # Define coarse regions (row slices on 64×64)
        regions = {
            "forehead":   diff[2:18,  10:54],
            "eyes":       diff[18:30, 8:56],
            "nose/cheeks": diff[28:44, 6:58],
            "mouth/chin": diff[44:60, 12:52],
        }
        region_means = {r: float(v.mean()) for r, v in regions.items()}

        fig_bar_diff = go.Figure(go.Bar(
            x=list(region_means.keys()),
            y=list(region_means.values()),
            marker_color=[CYAN, "#ff6b6b", CYAN, CYAN],
            showlegend=False,
        ))
        fig_bar_diff.update_layout(
            **PLOTLY_LAYOUT,
            yaxis_title="Mean |Δpixel|",
            title=f"Mean displacement per region  (φ{ef_index}, α={alpha:.1f}σ)",
        )
        st.plotly_chart(fig_bar_diff, use_container_width=True)

        if alpha != 0:
            eye_effect = region_means["eyes"]
            skin_effect = max(region_means["forehead"],
                              region_means["nose/cheeks"],
                              region_means["mouth/chin"])
            ratio = eye_effect / (skin_effect + 1e-9)
            msg = (
                f"Eyes affected **{ratio:.2f}×** as much as the most-changed skin region. "
            )
            if ratio < 0.6:
                msg += "φᵢ mostly changes skin luminance — eye whites are relatively preserved."
            elif ratio < 1.1:
                msg += "φᵢ affects eyes and skin roughly equally."
            else:
                msg += "φᵢ changes the eye region more than skin — not an illumination-only mode."
            st.info(msg)
