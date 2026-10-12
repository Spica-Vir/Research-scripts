#!/usr/bin/env python3

import argparse
import numpy as np

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


def main():
    parser = argparse.ArgumentParser(
        description="Apply strain to a VASP POSCAR file"
    )
    parser.add_argument("input", help="Input POSCAR file")
    parser.add_argument(
        "strain",
        nargs="+",
        type=float,
        help="Provide 1, 3, or 9 strain values",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="POSCAR_strained",
        help="Output POSCAR filename",
    )

    args = parser.parse_args()
    values = np.array(args.strain, dtype=float)

    if len(values) == 1:
        # Uniform strain
        strain_tensor = np.eye(3) * values[0]

    elif len(values) == 3:
        # Axial strains: exx, eyy, ezz
        strain_tensor = np.diag(values)

    elif len(values) == 9:
        # Full strain tensor, row by row
        strain_tensor = values.reshape((3, 3))

    else:
        parser.error("Provide exactly 1, 3, or 9 strain values.")

    structure = Structure.from_file(args.input)

    # Supported by older and newer pymatgen versions
    strained_structure = structure.copy()
    strained_structure.apply_strain(strain_tensor, inplace=True)

    Poscar(strained_structure, sort_structure=False).write_file(args.output)

    print(f"Strained POSCAR written to: {args.output}")


if __name__ == "__main__":
    main()
