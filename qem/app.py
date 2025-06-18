import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import imageio
import tempfile
import os
import hyperspy.api as hs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from qem.image_fitting import ImageModelFitting

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Quantitative Electron Microscopy (QEM)")

# --- Helper Functions ---
def initialize_session_state():
    """Initializes Streamlit session state variables."""
    defaults = {
        "original_image_data": None,
        "image_model_fitter": None,
        "pixel_size": 1.0,
        "current_model_name": "Gaussian",
        "atom_types_str": "",
        "unit_cell_str": "",
        "cif_file_path_state": None,
        "params_initialized": False,
        "active_plot": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_matplotlib_plot(fig, use_container_width=True):
    """Displays a Matplotlib figure in Streamlit."""
    if fig:
        st.session_state.active_plot = fig
        st.pyplot(fig, use_container_width=use_container_width)

def create_interactive_peak_plot(image, peaks=None, title="Interactive Peak Selection"):
    """Creates an interactive Plotly figure for peak selection."""
    # Normalize image for display
    if np.issubdtype(image.dtype, np.floating):
        min_v, max_v = np.percentile(image, 5), np.percentile(image, 95)
        img_display = np.clip((image - min_v) / (max_v - min_v), 0, 1)
    else:
        img_display = image
    
    fig = make_subplots()
    
    fig.add_trace(go.Heatmap(
        z=img_display,
        colorscale='gray',
        showscale=False
    ))
    
    if peaks is not None and len(peaks) > 0:
        fig.add_trace(go.Scatter(
            x=peaks[:, 0],
            y=peaks[:, 1],
            mode='markers',
            marker=dict(color='red', size=10, symbol='circle-open'),
            name='Peaks'
        ))
    
    fig.update_layout(
        title=title,
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False, scaleanchor='x'),
        hovermode='closest'
    )
    
    fig.add_annotation(
        text="Click to add/remove peaks, then click 'Confirm Peaks' when done",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False
    )

    return fig

def load_image_universal(file_source):
    """Loads image from uploaded file or path, supporting standard and Hyperspy formats."""
    image_data = None
    filename = None
    temp_file_path = None
    try:
        if hasattr(file_source, 'name'):  # Uploaded file object
            filename = file_source.name
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in ['.hspy', '.emd', '.dm3', '.dm4']:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    tmp.write(file_source.getbuffer())
                    temp_file_path = tmp.name
                signal = hs.load(temp_file_path)
                image_data = signal.data
            else:
                image_data = imageio.imread(file_source)
        elif isinstance(file_source, str): # File path string
            filename = file_source
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in ['.hspy', '.emd', '.dm3', '.dm4']:
                signal = hs.load(filename)
                image_data = signal.data
            else:
                image_data = imageio.imread(filename)
        if image_data is not None:
            st.success(f"File '{filename}' loaded successfully.")
        else:
            st.error("Invalid file source or failed to load.")
    except Exception as e:
        st.error(f"Error loading file '{filename}': {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return image_data

# --- UI Rendering Functions for Each Stage ---

def render_load_data_stage(main_area_col1, main_area_col2):
    """Renders the UI for the Load Data & Preprocessing stage."""
    with main_area_col1:
        st.header("1. Load Image")
        uploaded_file = st.file_uploader(
            "Drag & drop or click to upload (tif, png, hspy, emd, etc.)",
            type=["tif", "tiff", "png", "jpg", "jpeg", "hspy", "emd", "dm3", "dm4"]
        )
        file_path_input = st.text_input("Or enter full file path")

        if st.button("Load Image", key="load_image_btn"):
            file_to_load = uploaded_file if uploaded_file else file_path_input
            if file_to_load:
                with st.spinner("Loading image..."):
                    img_data = load_image_universal(file_to_load)
                    if img_data is not None:
                        st.session_state.original_image_data = img_data
                        st.session_state.image_model_fitter = None # Reset fitter on new image
                        st.session_state.params_initialized = False
            else:
                st.warning("Please provide a file to load.")

        if st.session_state.original_image_data is not None:
            st.header("2. Initialize Processor")
            # Define available model types directly based on ImageModelFitting implementation
            model_names = ["gaussian", "voigt", "lorentzian"]
            current_model_idx = 0
            if st.session_state.current_model_name in model_names:
                current_model_idx = model_names.index(st.session_state.current_model_name)
            st.session_state.current_model_name = st.selectbox("Select Model Type", model_names, index=current_model_idx)
            st.session_state.pixel_size = st.number_input("Pixel Size (e.g., Å/px)", value=st.session_state.pixel_size, min_value=0.001, format="%.4f")

            with st.expander("Optional Parameters (Atom Types, Unit Cell, CIF)"):
                st.session_state.atom_types_str = st.text_input("Atom Types (comma-separated, e.g., Sr,Ti,O)", value=st.session_state.atom_types_str)
                st.session_state.unit_cell_str = st.text_input("Unit Cell (a,b,c,α,β,γ; e.g. 3.9,3.9,3.9,90,90,90)", value=st.session_state.unit_cell_str)
                cif_file_upload = st.file_uploader("Upload CIF file (optional)", type=['cif'])
                if cif_file_upload:
                    st.session_state.cif_file_path_state = cif_file_upload.name
                    st.info(f"CIF file '{cif_file_upload.name}' ready to use.")

            if st.button("Initialize ImageModelFitter"): 
                if st.session_state.original_image_data is not None:
                    atom_types = [s.strip() for s in st.session_state.atom_types_str.split(',') if s.strip()] or None
                    unit_cell_list = None
                    if st.session_state.unit_cell_str:
                        try:
                            unit_cell_list = [float(x.strip()) for x in st.session_state.unit_cell_str.split(',')]
                            if len(unit_cell_list) != 6:
                                st.error("Unit cell must have 6 values.")
                                unit_cell_list = None
                        except ValueError:
                            st.error("Invalid format for unit cell parameters.")
                    with st.spinner("Initializing ImageModelFitter..."):
                        st.session_state.image_model_fitter = ImageModelFitting(
                            image=st.session_state.original_image_data,
                            dx=st.session_state.pixel_size,
                            units="A",  # Assuming pixel_size is in Angstroms from UI label
                            elements=atom_types if atom_types else None,
                            model_type=st.session_state.current_model_name
                        )
                        st.success("ImageModelFitting initialized successfully!")
                        st.session_state.params_initialized = False # Reset on new init
                else:
                    st.error("Failed to initialize ImageModelFitter. Ensure image is loaded and model is selected.")

        if st.session_state.image_model_fitter:
            st.header("3. Preprocessing Steps")
            fitter = st.session_state.image_model_fitter
            with st.expander("Peak Finding", expanded=False):
                pf_threshold = st.number_input("Threshold", value=0.1, format="%.3f", key="pf_thresh")
                pf_min_dist = st.number_input("Min Distance (px)", value=5, min_value=1, key="pf_min_dist")
                pf_sigma = st.number_input("Gaussian Sigma", value=5.0, min_value=0.1, format="%.1f", key="pf_sigma")
                
                # Initialize session state for peak finding
                if "peak_finding_active" not in st.session_state:
                    st.session_state.peak_finding_active = False
                if "current_peaks" not in st.session_state:
                    st.session_state.current_peaks = None
                if "temp_image" not in st.session_state:
                    st.session_state.temp_image = None
                
                # Button to initiate peak finding
                if st.button("Find Peaks", key="find_peaks_btn"):
                    with st.spinner("Finding peaks..."):
                        # Store filtered image for consistent peak finding
                        image_filtered = gaussian_filter(fitter.image, pf_sigma)
                        st.session_state.temp_image = image_filtered
                        
                        # Find peaks using skimage directly (similar to ImageModelFitting.find_peaks)
                        peaks_locations = peak_local_max(
                            image_filtered,
                            min_distance=pf_min_dist,
                            threshold_rel=pf_threshold,
                            exclude_border=False,
                        )
                        
                        # Store peaks in session state (y, x format for consistency)
                        st.session_state.current_peaks = peaks_locations[:, [1, 0]].astype(float)
                        st.session_state.peak_finding_active = True
                        
                        # Show success message
                        st.success(f"{len(st.session_state.current_peaks)} peaks found. Interactive plot updated in the right panel.")
                
                # Controls for confirming peaks remain here
                if st.session_state.peak_finding_active:
                    st.info("Click on the plot in the right panel to add or remove peaks. Click 'Confirm Peaks' below when you're done.")
                    if st.button("Confirm Peaks", key="confirm_peaks_btn"):
                        # Apply the peaks to the fitter
                        fitter.coordinates = st.session_state.current_peaks
                        fitter.atom_types = np.zeros(len(fitter.coordinates), dtype=int)
                        
                        # Reset peak finding state
                        st.session_state.peak_finding_active = False
                        
                        # Show confirmation and the final plot (will appear in main_area_col2 via active_plot)
                        st.success(f"Confirmed {len(fitter.coordinates)} peaks!")

                        # Generate matplotlib plot for consistency with other functions
                        plt.figure()
                        plt.imshow(fitter.image, cmap="gray")
                        plt.scatter(fitter.coordinates[:, 0], fitter.coordinates[:, 1], color="red", s=10)
                        plt.title(f"Confirmed {len(fitter.coordinates)} Peaks")
                        display_matplotlib_plot(plt.gcf()) # This updates st.session_state.active_plot

            with st.expander("Lattice Mapping", expanded=False):
                lm_method = st.selectbox("Method", ['fft', 'template_match'], key="lm_method") # Add other methods
                lm_blur = st.number_input("Blur (px)", value=1.0, format="%.2f", key="lm_blur")
                lm_threshold = st.number_input("Threshold", value=0.5, format="%.2f", key="lm_thresh_map")
                if st.button("Map Lattice", key="map_lattice_btn"):
                    with st.spinner("Mapping lattice..."):
                        fitter.map_lattice(method=lm_method, blur=lm_blur, threshold_rel=lm_threshold, plot=True)
                        display_matplotlib_plot(plt.gcf())
                        st.success("Lattice mapping complete.")

            with st.expander("Import/Export Coordinates", expanded=False):
                coord_file_upload = st.file_uploader("Import Coordinates (.npy)", type=['npy'], key="import_coords_upload")
                if coord_file_upload:
                    if coord_file_upload.name.endswith('.npy'):
                        coords = np.load(coord_file_upload)
                        fitter.import_coordinates(coords)
                        st.success(f"Imported {len(coords)} coordinates.")
                    else:
                        st.warning("Only .npy import is simplified here. Add txt/csv parsing.")

            if st.button("Export Coordinates (.npy)", key="export_coords_btn"):
                if fitter.coordinates is not None and len(fitter.coordinates) > 0:
                    np.save("exported_coordinates.npy", fitter.coordinates)
                    st.success("Coordinates exported to exported_coordinates.npy")
                else:
                    st.warning("No coordinates to export.")

            with st.expander("Add/Remove Peaks", expanded=False):
                st.info("Interactive peak addition/removal is a planned feature.")

            with st.expander("Calibration", expanded=False):
                cal_a = st.number_input("Lattice 'a' (optional, Å)", value=0.0, format="%.4f", key="cal_a")
                cal_b = st.number_input("Lattice 'b' (optional, Å)", value=0.0, format="%.4f", key="cal_b")
                if st.button("Calibrate", key="calibrate_btn"):
                    if cal_a > 0 and cal_b > 0:
                        with st.spinner("Calibrating..."):
                            fitter.calibrate(a=cal_a, b=cal_b)
                            st.success(f"Calibration complete. New pixel size: {fitter.dx:.4f} Å/px")
                    else:
                        st.warning("Provide valid lattice parameters 'a' and 'b' for calibration.")

    with main_area_col2:
        st.header("Image Viewer / Plot")
        fitter = st.session_state.get("image_model_fitter")

        # 1. Interactive Peak Finding Plot (Highest Priority if active)
        if st.session_state.get("peak_finding_active") and st.session_state.get("current_peaks") is not None:
            image_data_for_plot = None
            if st.session_state.temp_image is not None:
                image_data_for_plot = st.session_state.temp_image
            elif fitter and fitter.image is not None:
                image_data_for_plot = fitter.image
            elif st.session_state.original_image_data is not None:
                image_data_for_plot = st.session_state.original_image_data

            if image_data_for_plot is not None:
                fig = create_interactive_peak_plot(
                    image_data_for_plot,
                    st.session_state.current_peaks,
                    "Interactive Peak Selection - Edit Plot, Confirm Left"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Cannot display interactive peak plot: image data missing.")
        # 2. Other Active Matplotlib Plot (e.g., from lattice mapping, confirmed peaks)
        elif st.session_state.get("active_plot"):
            display_matplotlib_plot(st.session_state.active_plot)
        # 3. Original Loaded Image (if no specific plot is active)
        elif st.session_state.get("original_image_data") is not None:
            st.subheader("Original Image")
            img_display = st.session_state.original_image_data
            if np.issubdtype(img_display.dtype, np.floating): # Normalization
                min_v, max_v = np.min(img_display), np.max(img_display)
                img_display = (img_display - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(img_display)
            st.image(img_display, use_column_width=True)

            if fitter:
                st.info("Perform an action in the left panel to process or plot.")
            else:
                st.info("Initialize ImageModelFitter in the left panel to proceed.")
        # 4. Placeholder if nothing to show
        else:
            st.info("Load an image using the controls in the left panel to begin.")

def render_fitting_stage(main_area_col1, main_area_col2):
    """Renders the UI for the Fitting stage."""
    fitter = st.session_state.get("image_model_fitter")
    if not fitter:
        st.warning("Please load an image and initialize ImageModelFitter in the 'Load Data & Preprocessing' stage first.")
        return

    with main_area_col1:
        st.header("Fitting Parameters & Controls")

        if not st.session_state.params_initialized:
            if st.button("Initialize/Guess Parameters", key="init_guess_params_btn"):
                with st.spinner("Initializing parameters..."):
                    fitter.init_params(guess_params=True)
                    st.session_state.params_initialized = True
                    st.success("Parameters initialized.")
        else:
            st.info("Parameters already initialized.")

        with st.expander("Refine Center of Mass", expanded=False):
            if st.button("Refine CoM", key="refine_com_btn"):
                if not st.session_state.params_initialized:
                    st.warning("Initialize parameters first.")
                else:
                    with st.spinner("Refining Center of Mass..."):
                        fitter.refine_center_of_mass(plot=True)
                        display_matplotlib_plot(plt.gcf())
                        st.success("Center of Mass refinement complete.")

        with st.expander("Refine Local Max", expanded=False):
            rlm_window = st.number_input("Window Size (optional)", value=0, min_value=0, key="rlm_window")
            if st.button("Refine Local Max", key="refine_lm_btn"):
                if not st.session_state.params_initialized:
                    st.warning("Initialize parameters first.")
                else:
                    with st.spinner("Refining Local Maxima..."):
                        fitter.refine_local_max(window_size=rlm_window if rlm_window > 0 else None, plot=True)
                        display_matplotlib_plot(plt.gcf())
                        st.success("Local Maxima refinement complete.")

        st.markdown("--- Advanced Fitting ---")
        fit_maxiter = st.number_input("Max Iterations for Fitting", value=100, min_value=10, key="fit_maxiter")
        fit_tol = st.number_input("Tolerance for Fitting", value=1e-3, format="%.1e", key="fit_tol")

        if st.button("Fit Global", key="fit_global_btn"):
            if not st.session_state.params_initialized:
                st.warning("Initialize parameters first.")
            else:
                with st.spinner("Performing global fit..."):
                    fitter.fit_global(maxiter=fit_maxiter, tol=fit_tol)
                    display_matplotlib_plot(plt.gcf())
                    st.success("Global fit complete.")

        if st.button("Fit Voronoi", key="fit_voronoi_btn"):
            if not st.session_state.params_initialized:
                st.warning("Initialize parameters first.")
            else:
                with st.spinner("Performing Voronoi fit..."):
                    fitter.fit_voronoi(plot=True)
                    display_matplotlib_plot(plt.gcf())
                    st.success("Voronoi fit complete.")

    with main_area_col2:
        st.header("Fitting Output / Plot")
        if st.session_state.active_plot:
            display_matplotlib_plot(st.session_state.active_plot)
        else:
            st.info("Run a fitting or refinement method to see results.")
        
        if fitter.params is not None:
            st.subheader("Current Parameters")
            # Display a summary of parameters (can be extensive)
            param_summary = {k: (v.shape if hasattr(v, 'shape') else v) for k,v in fitter.params.items()}
            st.json(param_summary, expanded=False)

def render_analysis_stage(main_area_col1, main_area_col2):
    """Renders the UI for the Analysis stage (focus on ImageModelFitter methods)."""
    fitter = st.session_state.get("image_model_fitter")
    if not fitter or not st.session_state.params_initialized:
        st.warning("Please complete data loading, initialization, and fitting stages first.")
        return

    with main_area_col1:
        st.header("ImageModelFitter Analysis")
        with st.expander("Voronoi Integration", expanded=True):
            if st.button("Run Voronoi Integration", key="voronoi_int_btn"):
                with st.spinner("Performing Voronoi integration..."):
                    fitter.voronoi_integration(plot=True)
                    display_matplotlib_plot(plt.gcf())
                    st.success("Voronoi integration complete.")

        with st.expander("Scatter Channel Plots (SCS)", expanded=False):
            scs_layout = st.selectbox("Layout", ["horizontal", "vertical"], key="scs_layout")
            scs_per_element = st.checkbox("Per Element", key="scs_per_element")
            if st.button("Plot SCS", key="scs_plot_btn"):
                with st.spinner("Generating SCS plot..."):
                    fitter.plot_scs(layout=scs_layout, per_element=scs_per_element, plot=True)
                    display_matplotlib_plot(plt.gcf())
            if st.button("Plot SCS Histogram", key="scs_hist_btn"):
                with st.spinner("Generating SCS histogram..."):
                    fitter.plot_scs_histogram()
                    display_matplotlib_plot(plt.gcf())

        with st.expander("Volume Calculation", expanded=False):
            if st.button("Calculate Volume", key="calc_vol_btn"):
                with st.spinner("Calculating volume..."):
                    volume_data = fitter.volume()
                    st.json(volume_data)
                    st.success("Volume calculation complete.")

    with main_area_col2:
        st.header("Analysis Plot / Output")
        if st.session_state.active_plot:
            display_matplotlib_plot(st.session_state.active_plot)
        else:
            st.info("Run an analysis method to see results.")

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit application."""
    initialize_session_state()

    st.sidebar.title("QEM Analysis Suite")
    app_stage = st.sidebar.radio(
        "Select Stage",
        ["Data Loading & Pre-processing", "Fitting", "Analysis"],
        key="app_stage_selector"
    )

    # Create two main columns for the layout
    main_col1, main_col2 = st.columns([2, 3]) # Adjust ratio as needed, e.g., [2,1] or [1,2]

    if app_stage == "Data Loading & Pre-processing":
        render_load_data_stage(main_col1, main_col2)
    elif app_stage == "Fitting":
        render_fitting_stage(main_col1, main_col2)
    elif app_stage == "Analysis":
        render_analysis_stage(main_col1, main_col2)

if __name__ == "__main__":
    main()