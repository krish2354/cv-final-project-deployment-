"""Import the final Colab ZIP permanently for local or Streamlit deployment.

Usage: python prepare_artifacts.py /path/to/cellcount_colab_outputs.zip
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cellcount.artifacts import DETECTOR, SUPERRESOLUTION, import_colab_bundle, status


def main():
    parser=argparse.ArgumentParser(description='Stage your actual Colab ONNX weights and metrics into artifacts/.')
    parser.add_argument('zip_path',type=Path)
    args=parser.parse_args()
    if not args.zip_path.is_file():
        parser.error(f'No archive found at: {args.zip_path}')
    target = Path(__file__).resolve().parent / 'artifacts'
    files = import_colab_bundle(args.zip_path.read_bytes(),target)
    print(f'Imported {len(files)} Colab artifact(s) into {target}:')
    for name in sorted(files):
        f=target/name
        print(f'  {name:50} {f.stat().st_size/1_000_000:>9.2f} MB')
    current=status(target)
    print('\nReadiness:',json.dumps(current,indent=2))
    if not current['detector']:
        raise SystemExit('ERROR: trained YOLO ONNX is missing; rerun Colab export.')
    if not current['super_resolution']:
        raise SystemExit('WARNING: Real-ESRGAN ONNX is missing; GenAI condition cannot run.')
    print('\nBoth models are staged. Next: streamlit run app.py')


if __name__=='__main__':
    main()
