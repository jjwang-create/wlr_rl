from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ARCHIVE_DIR = Path(__file__).resolve().parent
CURVES_DIR = ARCHIVE_DIR / "curves"
PLOTS_DIR = ARCHIVE_DIR / "plots"

RUNS = {
    "model_602": ARCHIVE_DIR / "model_602" / "events.out.tfevents",
    "model_681": ARCHIVE_DIR / "model_681" / "events.out.tfevents",
}

PLOT_TAGS = {
    "mean_reward": ["Train/mean_reward", "mean_reward"],
    "episode_length": ["Train/mean_episode_length", "Episode/mean_episode_length", "mean_episode_length"],
    "value_loss": ["Loss/value_function", "Train/value_function_loss", "value_function_loss"],
    "surrogate_loss": ["Loss/surrogate", "Train/surrogate_loss", "surrogate_loss"],
    "action_noise": ["Train/mean_noise_std", "mean_noise_std"],
    "termination_timeout": ["Episode_Termination/time_out"],
    "termination_roll_pitch": ["Episode_Termination/roll_pitch"],
    "termination_non_wheel_contact": ["Episode_Termination/non_wheel_contact"],
    "front_clearance_reward": ["Episode_Reward/front_clearance"],
    "rear_clearance_reward": ["Episode_Reward/rear_clearance"],
    "flat_orientation_l2": ["Episode_Reward/flat_orientation_l2"],
    "lin_vel_xy_l2": ["Episode_Reward/lin_vel_xy_l2"],
}


def load_scalars(event_path: Path) -> dict[str, list[tuple[int, float]]]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    data: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        data[tag] = [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]
    return data


def write_long_csv(run_name: str, data: dict[str, list[tuple[int, float]]]) -> None:
    path = CURVES_DIR / f"{run_name}_scalars_long.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "tag", "step", "value"])
        for tag, values in sorted(data.items()):
            for step, value in values:
                writer.writerow([run_name, tag, step, value])


def select_tag(data: dict[str, list[tuple[int, float]]], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in data:
            return candidate
    return None


def write_summary_csv(all_data: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    path = CURVES_DIR / "standing_training_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "metric", "tag", "last_step", "last_value", "best_value"])
        for run_name, data in all_data.items():
            for metric, candidates in PLOT_TAGS.items():
                tag = select_tag(data, candidates)
                if tag is None:
                    continue
                values = data[tag]
                if not values:
                    continue
                last_step, last_value = values[-1]
                best_value = max(value for _, value in values)
                writer.writerow([run_name, metric, tag, last_step, last_value, best_value])


def plot_metric(all_data: dict[str, dict[str, list[tuple[int, float]]]], metric: str, candidates: list[str]) -> None:
    plt.figure(figsize=(9, 5))
    plotted = False
    for run_name, data in all_data.items():
        tag = select_tag(data, candidates)
        if tag is None:
            continue
        values = data[tag]
        if not values:
            continue
        steps = [step for step, _ in values]
        scalars = [value for _, value in values]
        plt.plot(steps, scalars, label=f"{run_name}: {tag}")
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.title(metric.replace("_", " ").title())
    plt.xlabel("Training iteration")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{metric}.png", dpi=160)
    plt.close()


def main() -> None:
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for run_name, event_path in RUNS.items():
        data = load_scalars(event_path)
        all_data[run_name] = data
        write_long_csv(run_name, data)

    write_summary_csv(all_data)
    for metric, candidates in PLOT_TAGS.items():
        plot_metric(all_data, metric, candidates)

    print(f"Wrote curves to {CURVES_DIR}")
    print(f"Wrote plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
