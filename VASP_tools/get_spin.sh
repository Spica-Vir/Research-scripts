#!/usr/bin/env bash

# Usage: ./get_spin.sh [OUTCAR_FILE] [RELAXATION_STEP] [SPIN_OUTPUT_FILE]
#
# Defaults:
#   OUTCAR_FILE     = OUTCAR
#   RELAXATION_STEP = -1 (latest step; -2 is the previous step, and so on)
#   SPIN_OUTPUT_FILE is empty, so no data file is written.
#
# Relaxation and SCF steps are printed using zero-based indices. For SCF-only
# calculations, RELAXATION_STEP is ignored. When SPIN_OUTPUT_FILE is given,
# the script writes a fixed-width, space-separated table containing:
#   SCF step, spin up, spin down, total spin, and net spin for each atom.
# Atom labels use the element and global one-based POSCAR index (for example,
# Co_9). Per-atom net spin is trace(spin up) - trace(spin down).
# LDAUPRINT = 1 is required for spin information and the optional data file.

set -o pipefail

outcar=${1:-OUTCAR}
requested_step=${2:--1}
spin_output=${3:-}

if [[ ! -r "$outcar" ]]; then
    printf 'Error: cannot read OUTCAR file: %s\n' "$outcar" >&2
    exit 1
fi

step_count=$(awk '/Ionic step[[:space:]]+[0-9]+/ { count++ } END { print count + 0 }' "$outcar")

if (( step_count == 0 )); then
    scf_only=1
    selected_step=0
    printf 'Relaxation step: not applicable (SCF-only output)\n'
else
    scf_only=0

    if [[ ! "$requested_step" =~ ^-?[0-9]+$ ]]; then
        printf 'Error: relaxation step must be an integer: %s\n' "$requested_step" >&2
        exit 2
    fi

    if (( requested_step < 0 )); then
        selected_step=$((step_count + requested_step))
    else
        selected_step=$requested_step
    fi

    if (( selected_step < 0 || selected_step >= step_count )); then
        printf 'Error: relaxation step %s is out of range (available: 0 to %d)\n' \
            "$requested_step" "$((step_count - 1))" >&2
        exit 4
    fi

    printf 'Relaxation step: %d\n' "$selected_step"
fi

awk -v wanted_ionic="$((selected_step + 1))" -v scf_only="$scf_only" \
    -v spin_output="$spin_output" '
BEGIN {
    if (scf_only)
        in_wanted = 1
}

function save_scf() {
    if (!have_iteration || scf_saved)
        return

    save_atom()
    if (scf_has_spin) {
        spin_up[scf_index] = scf_up
        spin_down[scf_index] = scf_down
        spin_seen[scf_index] = 1
        any_spin = 1
    }
    if (scf_index > max_scf)
        max_scf = scf_index
    scf_saved = 1
}

function save_atom(    key) {
    if (!atom_active || !atom_has_spin)
        return

    key = scf_index SUBSEP atom_index
    atom_spin[key] = atom_up - atom_down
    atom_seen[atom_index] = 1
    if (atom_index > max_atom)
        max_atom = atom_index
    atom_has_spin = 0
}

function write_spin_output(    i, atom, key) {
    printf "#%13s", "SCF_step" > spin_output
    printf "%14s%14s%14s", "spin_up", "spin_down", "total_spin" > spin_output
    for (atom = 1; atom <= max_atom; atom++) {
        if (atom_seen[atom])
            printf "%14s", atom_label[atom] > spin_output
    }
    printf "\n" > spin_output

    for (i = 0; i <= max_scf; i++) {
        if (!spin_seen[i])
            continue

        printf "%14d%14.4f%14.4f%14.4f", i, spin_up[i], spin_down[i], \
            spin_up[i] - spin_down[i] > spin_output
        for (atom = 1; atom <= max_atom; atom++) {
            if (atom_seen[atom]) {
                key = i SUBSEP atom
                printf "%14.4f", atom_spin[key] + 0 > spin_output
            }
        }
        printf "\n" > spin_output
    }
    close(spin_output)
}

function print_results(    i, difference) {
    if (force_found)
        printf "FORCES: max atom = %.6f eV/Angst, RMS = %.6f eV/Angst\n", force_max, force_rms
    else
        print "The selected relaxation step has not finished; FORCES: max is not available."

    if (!any_spin) {
        print "Warning: LDAUPRINT = 1 must be set for spin information"
        return
    }

    printf "%10s%14s%14s%14s\n", "SCF step", "spin up", "spin down", \
        "total spin"
    for (i = 0; i <= max_scf; i++) {
        if (spin_seen[i]) {
            difference = spin_up[i] - spin_down[i]
            printf "%10d%14.4f%14.4f%14.4f\n", \
                i, spin_up[i], spin_down[i], difference
        }
    }

    if (spin_output != "") {
        write_spin_output()
        printf "Atom-specific spin distribution written into %s\n", spin_output
    }
}

/^[[:space:]]*VRHFIN[[:space:]]*=/ && !species_ready {
    element = $0
    sub(/^.*=/, "", element)
    sub(/:.*/, "", element)
    gsub(/[[:space:]]/, "", element)
    species[++species_count] = element
    next
}

/ions per type[[:space:]]*=/ && !species_ready {
    species_ready = 1
    next
}

/Ionic step[[:space:]]+[0-9]+/ {
    ionic = $4 + 0
    if (in_wanted && ionic != wanted_ionic) {
        save_scf()
        print_results()
        in_wanted = 0
        exit
    }
    in_wanted = (ionic == wanted_ionic)
    next
}

!in_wanted { next }

/Iteration[[:space:]]+[0-9]+\([[:space:]]*[0-9]+\)/ {
    save_scf()

    inner = $4
    gsub(/[^0-9]/, "", inner)
    scf_index = (inner + 0) - 1
    have_iteration = 1
    scf_saved = 0
    scf_has_spin = 0
    scf_up = 0
    scf_down = 0
    component = 0
    atom_active = 0
    matrix_row = 0
    next
}

/^atom =[[:space:]]*[0-9]+/ && have_iteration {
    save_atom()
    atom_index = $3 + 0
    atom_type = $6 + 0
    if (species[atom_type] != "")
        atom_label[atom_index] = species[atom_type] "_" atom_index
    else
        atom_label[atom_index] = "atom_" atom_index
    atom_active = 1
    atom_has_spin = 0
    atom_up = 0
    atom_down = 0
    next
}

/^[[:space:]]*occupancies and eigenvectors/ && atom_active {
    save_atom()
    component = 0
    atom_active = 0
    next
}

/^[[:space:]]*spin component[[:space:]]+[12][[:space:]]*$/ && atom_active {
    component = $3 + 0
    matrix_row = 0
    matrix_size = 0
    next
}

component != 0 {
    if ($0 ~ /^[[:space:]]*[-+.0-9]/ && NF > 0) {
        if (matrix_size == 0)
            matrix_size = NF
        matrix_row++
        if (matrix_row <= matrix_size && NF == matrix_size) {
            if (component == 1)
                scf_up += $(matrix_row)
            else
                scf_down += $(matrix_row)

            if (component == 1)
                atom_up += $(matrix_row)
            else
                atom_down += $(matrix_row)
        }
        scf_has_spin = 1
        atom_has_spin = 1
        if (matrix_row >= matrix_size)
            component = 0
        next
    }

    if (matrix_row > 0)
        component = 0
}

/FORCES:[[:space:]]+max atom,[[:space:]]+RMS/ {
    force_max = $5 + 0
    force_rms = $6 + 0
    force_found = 1
}

END {
    if (in_wanted) {
        save_scf()
        print_results()
    }
}
' "$outcar"
