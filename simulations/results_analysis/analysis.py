import os
from simulations.utils.metrics import compute_selfconsumption, compute_from_grid, compute_community_transfer

base_path = "../data_ev"
home_path = os.path.join(base_path, "home_data")
workplace_path = os.path.join(base_path, "workplace_data")
trips_path = os.path.join(base_path, "trips_data")

# Output files
out_self = "../results/results_self_consumption.csv"
out_grid = "../results/results_from_grid.csv"


def detect_smart_flag(filename):
    if "noSM_noPublic" in filename:
        return "noSM_noPublic"
    elif "noSM" in filename:
        return "noSM"
    elif "oracle" in filename:
        return "oracle"
    elif "RMSE" in filename:
        return "SM_RMSE"
    elif "MAE" in filename:
        return "SM_MAE"
    elif "SM" in filename:
        return "SM"
    else:
        return "unknown"


for file in os.listdir(trips_path):
    if not file.endswith("_trips.csv") and not file.endswith("_trips_MAE.csv") and not file.endswith("_trips_RMSE.csv"):
        continue

    trips_file = os.path.join(trips_path, file)
    ev_name = file.replace("_trips.csv", "").replace("_trips_MAE.csv", "").replace("_trips_RMSE.csv", "")
    smart_flag = detect_smart_flag(file)

    home_file = os.path.join(home_path, file.replace("_trips", ""))
    workplace_file = os.path.join(workplace_path, file.replace("_trips", ""))

    if not (os.path.exists(home_file) and os.path.exists(workplace_file)):
        print(home_file, workplace_file)
        print(f"Missing data for {ev_name}. Skipping.")
        continue

    print(f"Processing {ev_name} ({smart_flag})...")

    try:
        compute_selfconsumption(ev_name, trips_file, smart_flag, out_self)
        compute_from_grid(ev_name, home_file, workplace_file, smart_flag, out_grid)
    except Exception as e:
        print(f"Error processing {ev_name}: {e}")
