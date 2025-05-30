import pandas as pd
import os
import re
import matplotlib as mpl
import seaborn as sns

mpl.use('TkAgg')
import matplotlib.pyplot as plt

mpl.rcParams.update({'font.size': 16})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Charter', 'XCharter', 'Georgia', 'Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
palette = [
    "#e05263", "#c18974", "#a1c084", "#659157", "#679A84", "#69a2b0", "#b4b6b1", "#ffcab1"
]

sns.set_palette(palette)

# Directories
input_dir = "../data_ev/trips_data"  # Adjust this to where your files are
output_dir = "../results/figures"
os.makedirs(output_dir, exist_ok=True)

# Regex pattern to extract info from filenames
pattern = r"EV_(\d+)_(\d+)_([a-zA-Z]+)(?:_([a-zA-Z]+))?_trips(?:_[A-Z]+)?\.csv"

self_consumption = pd.read_csv("../results/results_self_consumption.csv")


# Plot the results for the requested comparisons

# Self-consumption comparison (Average across all EVs per simulation type)
def plot_comparison(df):
    # Prettier labels
    pretty_labels = {
        "SM": "SC",
        "SM_MAE": "SC + MAE",
        "SM_RMSE": "SC + RMSE",
        "oracle": "Oracle SC",
        "noSM": "No SC",
        "noSM_noPublic": "No SC, No Public",
    }

    # Map pretty labels
    df['label'] = df['smart_charging'].map(pretty_labels).fillna(df['smart_charging'])

    # Define desired order
    label_order = ["No SC", "No SC, No Public", "Oracle SC", "SC", "SC + MAE", "SC + RMSE"]
    df = df[df['label'].isin(label_order)]  # Filter to those in the order
    df['label'] = pd.Categorical(df['label'], categories=label_order, ordered=True)
    df = df.sort_values('label')

    # Plot for Workplace
    plt.figure(figsize=(10, 6))
    plt.bar(df['label'], df['workplace'], color="#69a2b0")
    plt.ylabel('Average Self-Consumption (kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "self_consumption_comparison_workplace_avg.pdf"))
    plt.close()

    # Plot for Home
    plt.figure(figsize=(10, 6))
    plt.bar(df['label'], df['home'], color="#69a2b0")
    plt.ylabel('Average Self-Consumption (kWh)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "self_consumption_comparison_home_avg.pdf"))
    plt.close()


plot_comparison(self_consumption)


def stats_smart_charging(input_dir):
    # Updated regex pattern to extract simulation type from filenames
    pattern = r"EV_(\d+)_(\d+)_SM_trips(_(?:MAE|RMSE))?\.csv"

    # Initialize counters for the statistics
    total_ev_count = {"SM": 0, "SM_MAE": 0, "SM_RMSE": 0}
    total_ev_out_of_battery = {"SM": 0, "SM_MAE": 0, "SM_RMSE": 0}
    total_events_below_20 = {"SM": 0, "SM_MAE": 0, "SM_RMSE": 0}
    total_events_count = {"SM": 0, "SM_MAE": 0, "SM_RMSE": 0}

    # Iterate through all the files in the input directory
    for fname in os.listdir(input_dir):
        if not fname.endswith(".csv"):
            continue

        match = re.match(pattern, fname)
        if not match:
            continue

        # Extract simulation information from the filename
        ev_id, _, simu_suffix = match.groups()

        # Determine the simulation type based on the presence of MAE or RMSE
        if simu_suffix is None:
            simu_type = "SM"
        else:
            simu_type = "SM" + simu_suffix

        # Read the CSV file for the current EV
        try:
            df = pd.read_csv(os.path.join(input_dir, fname))

            # Track total number of EVs for each simulation type
            if simu_type in total_ev_count:
                total_ev_count[simu_type] += 1

            # Check if any EV runs out of battery (SoC <= 0 in the last row)
            last_row = df.iloc[-1]
            if last_row["SoC"] <= 0:
                total_ev_out_of_battery[simu_type] += 1

            # Check if any event has SoC < 20% (at any point during the trip)
            trips_below_20 = df[df["SoC"] < 0.2]
            total_events_below_20[simu_type] += len(trips_below_20)

            # Increment event count
            total_events_count[simu_type] += len(df)

        except Exception as e:
            print(f"Failed to process {fname}: {e}")
            continue

    # Compute and print the percentage of EVs running out of battery and events below 20% for each simulation
    stats = {}
    for simu_label in total_ev_count:
        # Avoid division by zero if no trips were processed for a simulation type
        ev_out_of_battery_pct = (total_ev_out_of_battery[simu_label] / 110) * 100 if \
            total_ev_count[simu_label] > 0 else 0

        events_below_20_pct = (total_events_below_20[simu_label] / total_events_count[simu_label]) * 100 if \
            total_events_count[simu_label] > 0 else 0

        stats[simu_label] = {
            "EV_out_of_battery_pct": round(ev_out_of_battery_pct, 2),
            "events_below_20_pct": round(events_below_20_pct, 2)
        }

    return stats


#stats = stats_smart_charging(input_dir)
#print(stats)


def compare_discharging_capacity(file_path):
    # Read the results_from_grid.csv file
    df = pd.read_csv(file_path)

    # Filter data for each simulation type
    sm_df = df[df['smart_charging'] == 'SM']
    sm_mae_df = df[df['smart_charging'] == 'SM_MAE']
    sm_rmse_df = df[df['smart_charging'] == 'SM_RMSE']

    # Calculate the total discharging capacity for each simulation type
    def calculate_discharging_capacity(df):
        # Sum across the relevant columns (ev_home, ev_workplace)
        total_discharging = df[['ev_home', 'ev_workplace']].sum().sum()
        return total_discharging

    # Calculate for each simulation type
    sm_discharging = calculate_discharging_capacity(sm_df)/110
    sm_mae_discharging = calculate_discharging_capacity(sm_mae_df)/110
    sm_rmse_discharging = calculate_discharging_capacity(sm_rmse_df)/110

    # Prepare a summary for comparison
    comparison = {
        "Simulation Type": ["SM", "SM_MAE", "SM_RMSE"],
        "Total Discharging Capacity": [sm_discharging, sm_mae_discharging, sm_rmse_discharging]
    }

    comparison_df = pd.DataFrame(comparison)
    return comparison_df


file_path = "../results/results_from_grid.csv"
discharging_comparison = compare_discharging_capacity(file_path)
print(discharging_comparison)


def plot_total_discharge_by_location(file_path, save_path_prefix='total_discharge'):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # List of the four smart charging types to include
    smart_types = ['SM', 'oracle', 'SM_MAE', 'SM_RMSE']

    # Prettier display labels
    pretty_labels = {
        "SM": "SC",
        "SM_MAE": "SC + MAE",
        "SM_RMSE": "SC + RMSE",
        "oracle": "Oracle SC"
    }
    display_labels = [pretty_labels.get(label, label) for label in smart_types]

    # Filter to include only the relevant smart charging types
    df_filtered = df[df['smart_charging'].isin(smart_types)]

    # Group by smart_charging type and sum values for each location
    grouped = df_filtered.groupby('smart_charging')[['ev_home', 'ev_workplace']].sum()

    # Sort to ensure a consistent order
    grouped = grouped.loc[smart_types]

    # Home plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(grouped.index)), grouped['ev_home'], color="#69a2b0")
    plt.xticks(ticks=range(len(display_labels)), labels=display_labels)
    plt.ylabel('Total Discharged Energy at Home (kWh/year)')
    plt.tight_layout()
    plt.savefig(f'../results/figures/{save_path_prefix}_home.pdf')

    # Workplace plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(grouped.index)), grouped['ev_workplace'], color="#69a2b0")
    plt.xticks(ticks=range(len(display_labels)), labels=display_labels)
    plt.ylabel('Total Discharged Energy at Workplace (kWh/year)')
    plt.tight_layout()
    plt.savefig(f'../results/figures/{save_path_prefix}_workplace.pdf')


plot_total_discharge_by_location("../results/results_from_grid.csv")


def stats_smart_charging_with_oracle(input_dir):
    # Extended pattern to capture all 6 simulation types
    pattern = r"EV_\d+_\d+_(noSM_noPublic_trips|noSM_trips|SM_oracle_trips|SM_trips(?:_(MAE|RMSE))?)\.csv"

    # Initialize counters for all 6 simulation types
    simu_types = [
        "noSM_noPublic_trips",
        "noSM_trips",
        "SM_trips",
        "SM_trips_MAE",
        "SM_trips_RMSE",
        "SM_oracle_trips"
    ]

    total_ev_count = {simu: 0 for simu in simu_types}
    total_ev_out_of_battery = {simu: 0 for simu in simu_types}
    total_events_below_20 = {simu: 0 for simu in simu_types}
    total_events_count = {simu: 0 for simu in simu_types}

    for fname in os.listdir(input_dir):
        if not fname.endswith(".csv"):
            continue

        match = re.match(pattern, fname)
        if not match:
            continue

        simu_type = match.group(1)

        try:
            df = pd.read_csv(os.path.join(input_dir, fname))

            total_ev_count[simu_type] += 1

            # Check if EV ends with SoC <= 0
            if df.iloc[-1]["SoC"] <= 0:
                total_ev_out_of_battery[simu_type] += 1

            # Count events with SoC < 20%
            total_events_below_20[simu_type] += (df["SoC"] < 0.2).sum()

            # Count total events
            total_events_count[simu_type] += len(df)

        except Exception as e:
            print(f"Failed to process {fname}: {e}")
            continue

    # Summarize stats
    stats = {}
    for simu_type in simu_types:
        ev_pct = (total_ev_out_of_battery[simu_type] / 110) * 100 if total_ev_count[simu_type] > 0 else 0
        event_pct = (total_events_below_20[simu_type] / total_events_count[simu_type]) * 100 if total_events_count[
                                                                                                    simu_type] > 0 else 0
        stats[simu_type] = {
            "EV_out_of_battery_%": round(ev_pct, 2),
            "events_below_20_%": round(event_pct, 2)
        }

    return stats


stats = stats_smart_charging_with_oracle(input_dir)
print(stats["SM_trips_RMSE"])
print(stats["SM_oracle_trips"])


def plot_ev_out_of_battery(stats_dict, output_dir="../results/figures", filename="ev_out_of_battery_comparison.pdf"):
    # Define the canonical order
    sort_order = [
        'noSM_noPublic_trips',  # No SC, no public charging
        'noSM_trips',  # No SC, with public charging
        'SM_trips',  # SC
        'SM_trips_MAE',  # SC + MAE
        'SM_trips_RMSE',  # SC + RMSE
        'SM_oracle_trips'  # Oracle SC
    ]

    # Prettier display names
    pretty_labels = {
        'noSM_noPublic_trips': 'No SC, No Public',
        'noSM_trips': 'No SC',
        'SM_trips': 'SC',
        'SM_trips_MAE': 'SC + MAE',
        'SM_trips_RMSE': 'SC + RMSE',
        'SM_oracle_trips': 'Oracle SC'
    }

    # Filter existing keys in stats_dict
    labels = [label for label in sort_order if label in stats_dict]
    percentages = [stats_dict[label]['EV_out_of_battery_%'] for label in labels]
    display_labels = [pretty_labels.get(label, label) for label in labels]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(display_labels, percentages, color="#69a2b0")
    plt.ylabel('% of EVs that ran out of battery')
    plt.ylim(0, max(percentages) * 1.2)
    plt.tight_layout()

    # Save
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


stats = stats_smart_charging_with_oracle(input_dir)
plot_ev_out_of_battery(stats)

def compute_soc_event_bands(input_dir):
    pattern = r"EV_\d+_\d+_(noSM_noPublic_trips|noSM_trips|SM_oracle_trips|SM_trips(?:_(MAE|RMSE))?)\.csv"

    simu_types = [
        "noSM_noPublic_trips",
        "noSM_trips",
        "SM_trips",
        "SM_trips_MAE",
        "SM_trips_RMSE",
        "SM_oracle_trips"
    ]

    bands = {
        "low": {},  # [1,6[
        "mid": {},  # [6,16[
        "high": {}  # [16,20[
    }

    for simu_type in simu_types:
        bands["low"][simu_type] = 0
        bands["mid"][simu_type] = 0
        bands["high"][simu_type] = 0

    total_events = {simu_type: 0 for simu_type in simu_types}

    for fname in os.listdir(input_dir):
        if not fname.endswith(".csv"):
            continue

        match = re.match(pattern, fname)
        if not match:
            continue

        simu_type = match.group(1)
        file_path = os.path.join(input_dir, fname)

        try:
            df = pd.read_csv(file_path)
            total_events[simu_type] += len(df)

            below_20 = df[df["SoC"] < 0.2]["SoC"] * 100

            bands["low"][simu_type] += ((below_20 >= 1) & (below_20 < 6)).sum()
            bands["mid"][simu_type] += ((below_20 >= 6) & (below_20 < 16)).sum()
            bands["high"][simu_type] += ((below_20 >= 16) & (below_20 < 20)).sum()

        except Exception as e:
            print(f"Failed to process {fname}: {e}")
            continue

    # Convert to percentages
    results = {
        "simu_types": simu_types,
        "low": [round((bands["low"][s] / total_events[s]) * 100, 2) if total_events[s] > 0 else 0 for s in simu_types],
        "mid": [round((bands["mid"][s] / total_events[s]) * 100, 2) if total_events[s] > 0 else 0 for s in simu_types],
        "high": [round((bands["high"][s] / total_events[s]) * 100, 2) if total_events[s] > 0 else 0 for s in simu_types]
    }

    return results


def grid_consumption_percentage_reduction_separate():
    # Load data
    df = pd.read_csv("../results/results_from_grid.csv")

    # Define labels
    strategy_map = {
        "noSM_noPublic": "No SC, No Public",
        "noSM": "No SC",
        "SM": "SC",
        "SM_MAE": "SC + MAE",
        "SM_RMSE": "SC + RMSE",
        "oracle": "Oracle SC"
    }
    df["label"] = df["smart_charging"].map(strategy_map)

    # Exclude "No SC, No Public" strategy
    df = df[df["label"] != "No SC, No Public"]

    # Helper function to process and plot each location
    def process_and_plot(location_col, location_name):
        # Aggregate average grid consumption per month and strategy
        agg = df.groupby(["month", "label"])[[location_col]].mean().reset_index()

        # Values are negative, convert to positive
        agg["total_grid"] = -agg[location_col]

        # Pivot for easier calculation
        pivot = agg.pivot(index="month", columns="label", values="total_grid")

        baseline_col = "No SC"
        reduction_pct = 100 * (pivot[baseline_col].values.reshape(-1, 1) - pivot) / pivot[baseline_col].values.reshape(
            -1, 1)
        reduction_pct = pd.DataFrame(reduction_pct, index=pivot.index, columns=pivot.columns)

        # Drop baseline column (0% reduction vs itself)
        reduction_pct = reduction_pct.drop(columns=[baseline_col])

        # Plot
        plt.figure(figsize=(10, 5))
        reduction_pct.plot(kind="bar", ax=plt.gca(), colormap="Set2")
        plt.ylabel(f"Grid Consumption Reduction(%)-{location_name}")
        plt.xlabel("Month")
        plt.xticks(rotation=0)
        plt.axhline(0, linestyle='--', color='gray')
        plt.legend(title="Charging Strategy")
        plt.tight_layout()

        # Save results
        os.makedirs("../results/figures", exist_ok=True)
        plt.savefig(f"../results/figures/monthly_grid_reduction_percent_{location_name.lower()}.pdf")
        plt.close()

        # Save CSV
        reduction_pct.to_csv(f"../results/figures/monthly_grid_reduction_percent_{location_name.lower()}.csv")

    # Run for home and workplace
    process_and_plot("grid_home", "Home")
    process_and_plot("grid_workplace", "Workplace")


grid_consumption_percentage_reduction_separate()

# Load the CSV file
df = pd.read_csv("../data_REC/home/data.csv", parse_dates=["datetime"])
df['energy'] = pd.to_numeric(df['energy'], errors='coerce')
grid_consumption = df[df['energy'] < 0]['energy'].abs().sum()
injection = df[df['energy'] > 0]['energy'].sum()
print(f"Total grid consumption of the community: {grid_consumption:.3f} kWh, total injection: {injection:.3f} kWh")
