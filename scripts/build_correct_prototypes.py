import sys


def main():
    message = (
        "build_correct_prototypes.py has been deprecated.\n"
        "The current StudentActionError model no longer uses a correct-action prototype bank.\n"
        "Use the kinematic-chain training pipeline instead."
    )
    print(message, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
