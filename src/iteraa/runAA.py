"""Standalone script for running Archetypal Analysis on a data subset.

This script is intended to be called by HPC job schedulers.
"""

import argparse

from iteraa.piaa import runAA


def main():
    """Main entry point for the iteraa-run-AA console script."""
    parser = argparse.ArgumentParser(description="Run Archetypal Analysis on a data subset.")
    parser.add_argument("fName", type=str, help="Path to subset pickle file")
    parser.add_argument("nArchetypes", type=int, help="Number of archetypes")
    parser.add_argument("outputsPicklesPath", type=str, help="Path to save output pickles")
    parser.add_argument("splitKeyword", type=str, default="data", help="Keyword for data subsets")
    parser.add_argument("--robust", action="store_true", help="Use robust AA")
    parser.add_argument("--tolerance", type=float, default=0.001, help="Convergence tolerance")
    parser.add_argument("--onlyZ", action="store_true", help="Only return Z matrix")

    args = parser.parse_args()

    runAA(
        fName=args.fName,
        nArchetypes=args.nArchetypes,
        outputsPicklesPath=args.outputsPicklesPath,
        splitKeyword=args.splitKeyword,
        robust=args.robust,
        tolerance=args.tolerance,
        onlyZ=args.onlyZ,
    )


if __name__ == "__main__":
    main()
