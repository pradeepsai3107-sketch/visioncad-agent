# VisionCAD Agent — quick demo

## What this prototype does
1. Upload an image (or use the built-in demo shape).
2. Extract outlines using OpenCV.
3. Simplify the detected contours into vector geometry.
4. Export an editable 2D DXF file that can be opened in AutoCAD, LibreCAD, or other DXF-compatible CAD tools.
5. Optionally ask Gemini to describe the image if an API key is configured.

## Run on Windows (VS Code terminal)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Optional Gemini description
Set the API key in the same terminal before launching Streamlit:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
streamlit run app.py
```

Do not put your API key in source code or show it during the presentation.

## Best input images for a reliable demo
- Black outline on a white background
- Simple mechanical sketch, logo, silhouette, or front-view object
- Minimal shadows and clutter
- Use the built-in demo shape first to verify the full pipeline

## Honest limitations to mention
- This MVP vectorizes visible 2D contours; it does not recover hidden surfaces or produce a true 3D solid.
- A single photo does not provide reliable physical scale or depth without calibration/reference measurements.
- Photos with texture, shadows, and complex backgrounds can create noisy contours.
- The DXF contains editable polylines, not a fully constrained parametric CAD model.

## Suggested live demo
1. Launch the app and use the built-in demo to verify that export works.
2. Upload a clean object outline.
3. Show the green contour overlay.
4. Download `visioncad_output.dxf`.
5. Open the DXF in a CAD viewer and explain how this is the first geometry-reconstruction module in a larger vision-agent architecture.
