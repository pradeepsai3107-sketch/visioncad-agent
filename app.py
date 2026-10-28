
import os
from io import BytesIO, StringIO

import cv2
import ezdxf
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(
    page_title="VisionCAD Agent",
    page_icon="◈",
    layout="wide",
)

# -------------------- UI STYLING --------------------

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}
.hero {
    padding: 1.25rem 1.5rem;
    border: 1px solid rgba(128,128,128,.3);
    border-radius: 16px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("◈ VisionCAD Agent")
st.caption(
    "Image understanding · Geometry extraction · 2D CAD generation"
)

st.markdown(
    '<div class="hero"><b>Prototype workflow</b><br>'
    'Upload an image or technical sketch. The vision pipeline '
    'extracts visible outlines, converts them into vector polylines, '
    'and exports an editable DXF drawing.</div>',
    unsafe_allow_html=True,
)

# -------------------- SIDEBAR SETTINGS --------------------

with st.sidebar:
    st.header("Extraction settings")

    threshold_mode = st.selectbox(
        "Image processing",
        [
            "Auto edges",
            "Dark lines on light background",
            "Light object on dark background",
        ],
    )

    blur = st.select_slider(
        "Noise reduction",
        options=[1, 3, 5, 7],
        value=3,
    )

    min_area = st.slider(
        "Ignore tiny contours",
        min_value=10,
        max_value=3000,
        value=120,
        step=10,
    )

    simplify = st.slider(
        "Contour simplification",
        min_value=1,
        max_value=20,
        value=5,
        help="Higher values create fewer vertices.",
    )

    scale_mm = st.number_input(
        "Approximate image width in mm",
        min_value=1.0,
        max_value=10000.0,
        value=100.0,
        step=10.0,
    )

    use_gemini = st.checkbox(
        "Add Gemini image description (optional)",
        value=False,
    )

    st.caption(
        "Scale is estimated from the width you enter. "
        "It is not a measured physical dimension."
    )

# -------------------- IMAGE INPUT --------------------

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp", "bmp"],
)

demo = st.toggle(
    "Use built-in demo shape instead",
    value=False,
)


def make_demo():
    """Create a simple test drawing."""
    canvas = np.full((500, 700, 3), 255, dtype=np.uint8)

    cv2.rectangle(
        canvas, (170, 130), (510, 370), (0, 0, 0), 5
    )

    cv2.circle(
        canvas, (340, 250), 65, (0, 0, 0), 5
    )

    cv2.rectangle(
        canvas, (220, 175), (280, 225), (0, 0, 0), 4
    )

    return Image.fromarray(canvas)


def image_to_array(image):
    """Convert a PIL image to an OpenCV BGR array."""
    return cv2.cvtColor(
        np.array(image.convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )


# -------------------- IMAGE ANALYSIS --------------------

def extract_contours(
    bgr,
    mode,
    blur_size,
    min_contour_area,
    epsilon_pct,
):
    """Detect and simplify outlines in an image."""

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if blur_size > 1:
        gray = cv2.GaussianBlur(
            gray,
            (blur_size, blur_size),
            0,
        )

    if mode == "Auto edges":
        edges = cv2.Canny(gray, 50, 150)

        kernel = np.ones((3, 3), np.uint8)

        processed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1,
        )

    elif mode == "Dark lines on light background":
        _, processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    else:
        _, processed = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    contours, _ = cv2.findContours(
        processed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    kept = []

    for contour in contours:
        area = abs(cv2.contourArea(contour))
        perimeter = cv2.arcLength(contour, True)

        if area < min_contour_area or perimeter < 15:
            continue

        epsilon = max(
            0.5,
            epsilon_pct / 100.0 * perimeter,
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )

        points = approx.reshape(-1, 2)

        if len(points) >= 2:
            kept.append((points, area))

    kept.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return processed, kept


# -------------------- CAD GENERATION --------------------

def create_dxf(
    contours,
    image_width_px,
    image_height_px,
    width_mm,
):
    """Convert image contours into an editable DXF drawing."""

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM

    msp = doc.modelspace()

    doc.layers.new(
        "IMAGE_OUTLINES",
        dxfattribs={"color": 7},
    )

    doc.layers.new(
        "CENTER_ORIGIN",
        dxfattribs={"color": 1},
    )

    scale = width_mm / max(image_width_px, 1)

    for points, _area in contours:
        cad_points = []

        for x, y in points:
            cad_x = float(x * scale)

            # Flip the vertical axis from image to CAD coordinates.
            cad_y = float((image_height_px - y) * scale)

            cad_points.append((cad_x, cad_y))

        if len(cad_points) >= 2:
            msp.add_lwpolyline(
                cad_points,
                close=True,
                dxfattribs={"layer": "IMAGE_OUTLINES"},
            )

    # Add a small origin marker.
    cross = 2.0

    msp.add_line(
        (-cross, 0),
        (cross, 0),
        dxfattribs={"layer": "CENTER_ORIGIN"},
    )

    msp.add_line(
        (0, -cross),
        (0, cross),
        dxfattribs={"layer": "CENTER_ORIGIN"},
    )

    # DXF is text-based: use StringIO, then encode for download.
    stream = StringIO()
    doc.write(stream)

    return stream.getvalue().encode("utf-8")


# -------------------- VISUAL PREVIEW --------------------

def render_preview(bgr, contours):
    """Draw detected outlines in green over the input image."""

    preview = bgr.copy()

    contour_arrays = [
        points.astype(np.int32).reshape(-1, 1, 2)
        for points, _area in contours
    ]

    cv2.drawContours(
        preview,
        contour_arrays,
        -1,
        (0, 180, 70),
        2,
    )

    return cv2.cvtColor(
        preview,
        cv2.COLOR_BGR2RGB,
    )


# -------------------- SELECT IMAGE --------------------

if demo:
    image = make_demo()

elif uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error(
            "Could not read this image. Try another PNG or JPG file."
        )
        st.stop()

else:
    image = None


if image is None:
    st.info(
        "Upload a high-contrast image, technical sketch, logo, "
        "or simple object silhouette — or enable the built-in demo."
    )
    st.stop()


# -------------------- RUN THE PIPELINE --------------------

bgr = image_to_array(image)

try:
    processed, contours = extract_contours(
        bgr,
        threshold_mode,
        blur,
        min_area,
        simplify,
    )

except cv2.error as exc:
    st.error(f"Image processing failed: {exc}")
    st.stop()


if not contours:
    st.warning(
        "No usable outlines were detected. Try a clearer image "
        "or lower the 'Ignore tiny contours' setting."
    )

    st.image(
        image,
        caption="Input image",
        use_container_width=True,
    )

    st.stop()


try:
    dxf_bytes = create_dxf(
        contours,
        bgr.shape[1],
        bgr.shape[0],
        scale_mm,
    )

except Exception as exc:
    st.error(f"CAD generation failed: {exc}")
    st.stop()


preview = render_preview(bgr, contours)


# -------------------- DISPLAY RESULTS --------------------

left, right = st.columns(2)

with left:
    st.subheader("Input image")

    st.image(
        image,
        use_container_width=True,
    )

with right:
    st.subheader("Detected geometry")

    st.image(
        preview,
        caption="Green lines show extracted contours",
        use_container_width=True,
    )


m1, m2, m3 = st.columns(3)

m1.metric(
    "Contours exported",
    len(contours),
)

m2.metric(
    "Total vertices",
    sum(len(points) for points, _ in contours),
)

m3.metric(
    "Estimated drawing width",
    f"{scale_mm:g} mm",
)


st.download_button(
    "⬇ Download editable 2D CAD (.dxf)",
    data=dxf_bytes,
    file_name="visioncad_output.dxf",
    mime="application/dxf",
    type="primary",
    use_container_width=True,
)


with st.expander("Image analysis details", expanded=True):
    st.write(
        f"- Image dimensions: **{bgr.shape[1]} × "
        f"{bgr.shape[0]} pixels**"
    )

    st.write(
        f"- Processing method: **{threshold_mode}**"
    )

    st.write(
        "- Contour extraction: **OpenCV + polygon simplification**"
    )

    st.write(
        "- CAD output: **DXF with editable lightweight polylines**"
    )

    st.write(
        "- Scale: **estimated from the entered image width; "
        "not inferred from the image itself**"
    )

    st.warning(
        "This prototype vectorizes visible contours. It does not "
        "infer hidden geometry or produce a fully constrained 3D model."
    )


# -------------------- OPTIONAL GEMINI ANALYSIS --------------------

if use_gemini:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.warning(
            "Gemini is optional. Set GEMINI_API_KEY in your "
            "terminal environment and restart Streamlit."
        )

    else:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            buffer = BytesIO()
            image.save(buffer, format="JPEG")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    (
                        "Analyze this image for an engineering prototype. "
                        "Describe visible objects, main geometric primitives, "
                        "symmetry, and what a 2D outline can and cannot "
                        "represent. Do not invent dimensions."
                    ),
                    types.Part.from_bytes(
                        data=buffer.getvalue(),
                        mime_type="image/jpeg",
                    ),
                ],
            )

            st.subheader("AI visual interpretation")
            st.write(response.text)

        except Exception as exc:
            st.error(f"Gemini request failed: {exc}")

            st.caption(
                "The OpenCV contour-to-DXF pipeline works independently "
                "of Gemini."
            )
