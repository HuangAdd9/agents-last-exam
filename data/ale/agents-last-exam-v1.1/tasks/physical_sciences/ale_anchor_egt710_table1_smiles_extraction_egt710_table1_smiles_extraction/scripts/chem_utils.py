"""Small dependency-free SMILES bookkeeping helpers used by the oracle.

This is deliberately not a SMILES parser.  It only computes transparent,
token-level invariants (element-token counts and explicit stereochemical marks)
after the evaluator's exact canonical-SMILES gate has run.
"""
from __future__ import annotations

import re


def element_tokens(smiles: str) -> list[str]:
    # Scan atom tokens, including lower-case aromatic atoms.  This is a
    # bookkeeping tokenizer rather than a full SMILES parser.
    out: list[str] = []
    i = 0
    aromatic = set("bcnops")
    # Only consume a lower-case character after an upper-case atom when the
    # pair is a real two-letter element used by the staged records.  Treating
    # the `c` in `Nc1...` as part of a fictional `Nc` atom silently drops an
    # aromatic carbon from every such molecule.
    two_letter = {"Cl", "Br", "Si", "Se", "Na", "Li", "Al", "Ca", "Mg"}
    while i < len(smiles):
        ch = smiles[i]
        if ch == "[":
            j = smiles.find("]", i + 1)
            if j < 0:
                break
            m = re.match(r"([A-Z][a-z]?|[bcnops])", smiles[i + 1 : j])
            if m:
                out.append(m.group(1))
            i = j + 1
            continue
        if ch.isupper():
            pair = smiles[i : i + 2]
            if pair in two_letter:
                out.append(pair); i += 2; continue
            out.append(ch); i += 1; continue
        if ch in aromatic:
            out.append(ch); i += 1; continue
        i += 1
    return out


def bookkeeping(smiles: str) -> dict[str, int]:
    elems = element_tokens(smiles)
    hetero = sum(e.lower() != "c" for e in elems)
    return {
        "heavy_atoms": len(elems),
        "hetero_atoms": hetero,
        "fluorine_atoms": sum(e.upper() == "F" for e in elems),
        "chiral_marks": smiles.count("@"),
    }
