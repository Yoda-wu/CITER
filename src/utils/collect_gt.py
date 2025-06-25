import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser(
        description="Collect ground truth data from distributed results."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the distributed input data.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the collected output data.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of process to use.",
    )

    args = parser.parse_args()

    files = os.listdir(args.input_path)
    files = [f for f in files if f.endswith(".pkl")]
    files = sorted(files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    ranks = sorted([int(f.split("_")[-1].split(".")[0]) for f in files])
    assert ranks == list(range(len(ranks))) == list(range(args.world_size)), (
        "Distribution is not complete, ranks are missing. ranks: {ranks}"
    )

    data = []
    for file in files:
        with open(os.path.join(args.input_path, file), "rb") as f:
            data.extend(pickle.load(f))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Ground truth data collected and saved to {args.output_path}.")


if __name__ == "__main__":
    main()
