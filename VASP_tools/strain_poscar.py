#!/usr/bin/env python3

import argparse
import numpy as np

from pymatgen.core import Structure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input POSCAR")
    parser.add_argument("strain", nargs="+", type=float)
    parser.add_argument("-o", "--output", default="POSCAR_strained")
    args = parser.parse_args()

    values = np.asarray(args.strain, dtype=float)

    if len(values) == 1:
        strain_tensor = np.eye(3) * values[0]
    elif len(values) == 3:
        strain_tensor = np.diag(values)
    elif len(values) == 9:
        strain_tensor = values.reshape(3, 3)
    else:
        parser.error("Provide exactly 1, 3, or 9 strain values.")

    # Read the structure with pymatgen and apply the strain
    structure = Structure.from_file(args.input)
    strained_structure = structure.copy()
    strained_structure.apply_strain(strain_tensor, inplace=True)

    # Read the original POSCAR text.
    # This preserves duplicate species blocks such as Co Co.
    with open(args.input, "r") as f:
        lines = f.readlines()

    if len(lines) < 5:
        raise ValueError("Input file is not a valid POSCAR.")

    # POSCAR lattice vectors occupy lines 3--5.
    # Set scale to 1.0 because the vectors below are absolute.
    lines[1] = "1.0\n"

    lattice = strained_structure.lattice.matrix

    for i in range(3):
        lines[2 + i] = (
            f"{lattice[i, 0]:20.16f}"
            f"{lattice[i, 1]:20.16f}"
            f"{lattice[i, 2]:20.16f}\n"
        )

    with open(args.output, "w") as f:
        f.writelines(lines)

    print(f"Strained POSCAR written to: {args.output}")


if __name__ == "__main__":
    main()

