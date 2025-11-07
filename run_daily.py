import modal

# ----- Config -----
REPO_URL = "https://github.com/cimun/mlfs-book"
NB_FEATURE = "notebooks/airquality/2_air_quality_feature_pipeline.ipynb"
NB_INFER  = "notebooks/airquality/4_air_quality_batch_inference.ipynb"

# If you have per-run parameters (e.g., sensor/city), add here:
FEATURE_PARAMS = {}  # e.g., {"city": "stockholm", "sensor_id": "xxx"}
INFER_PARAMS   = {}  # e.g., {"forecast_horizon_days": 7}

# Name of your Modal Secret containing HOPSWORKS_API_KEY (and optional others)
HOPSWORKS_SECRET = "hopsworks"

# ----- Modal setup -----
stub = modal.App("air_quality_pipelines")

# Build an image with the tools we need.
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install([
        # core
        "hopsworks", "pandas", "numpy", "requests",
        # training/inference deps if used in notebooks
        "scikit-learn", "xgboost", "joblib", "matplotlib",
        # notebook execution
        "papermill", "ipykernel",
    ])
)

def _clone_repo(tempdir: str):
    import subprocess
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tempdir], check=True)

def _exec_notebook(nb_path: str, out_path: str, parameters: dict):
    import papermill as pm
    pm.execute_notebook(
        nb_path,
        out_path,
        parameters=parameters or {},
        kernel_name="python3",
        progress_bar=False,
        log_output=True,
    )

@stub.function(
    image=image, 
    schedule=modal.Period(days=1),     # runs daily
    secrets=[modal.Secret.from_name(HOPSWORKS_SECRET)],
    timeout=60 * 20,                    # bump if needed
)
def run_daily_features():
    import os, tempfile, pathlib
    workdir = tempfile.mkdtemp()
    _clone_repo(workdir)
    nb_in  = os.path.join(workdir, NB_FEATURE)
    nb_out = os.path.join(workdir, "out_feature.ipynb")

    # optional: pass parameters like sensor/city here
    _exec_notebook(nb_in, nb_out, FEATURE_PARAMS)

    # if the notebook writes artifacts (PNGs/CSV) into the repo,
    # you can push them somewhere here (e.g., S3, GH Pages). Otherwise, done.

@stub.function(
    image=image,
    # optional: run a little later than features (e.g., 25h cadence or a cron)
    schedule=modal.Period(days=1),
    secrets=[modal.Secret.from_name(HOPSWORKS_SECRET)],
    timeout=60 * 20,
)
def run_daily_inference():
    import os, tempfile
    workdir = tempfile.mkdtemp()
    _clone_repo(workdir)
    nb_in  = os.path.join(workdir, NB_INFER)
    nb_out = os.path.join(workdir, "out_infer.ipynb")

    _exec_notebook(nb_in, nb_out, INFER_PARAMS)

    # If your inference notebook outputs a PNG dashboard,
    # optionally commit/publish it from here (see note below).

if __name__ == "__main__":
    # Deploy both scheduled functions
    stub.deploy()
    # Run once interactively if you want to test now:
    with stub.run():
        run_daily_features.call()
        run_daily_inference.call()
