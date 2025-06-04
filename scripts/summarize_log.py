#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

# Add parent directory to Python path to find utils module
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.utils import parse_nccl_log

def print_summary(log_file_path: str):
    """Print a summary of the NCCL test log file."""
    try:
        data = parse_nccl_log(log_file_path)

        print(f"NCCL Test Log Summary: {Path(log_file_path).name}")
        print("=" * 50)

        # Print metadata
        print(f"Job ID: {data.get('jobid', 'N/A')}")
        print(f"NCCL Version: {data.get('nccl_version', 'N/A')}")
        print(f"Number of Nodes: {data.get('num_nodes', 'N/A')}")
        print(f"Number of GPUs: {data.get('num_gpus', 'N/A')}")
        print(f"Uses Alt Read: {data.get('uses_alt_read', False)}")
        print(f"Average Bus Bandwidth: {data.get('avg_bus_bandwidth', 'N/A')} GB/s")

        # Print performance data summary
        perf_data = data.get('performance_data')
        if perf_data:
            print(f"\nPerformance Data: {len(perf_data)} test cases")
            print("-" * 30)

            # Find max bandwidth entries
            max_oop_busbw = max(perf_data, key=lambda x: x['oop_busbw_gbps'])
            max_ip_busbw = max(perf_data, key=lambda x: x['ip_busbw_gbps'])

            print(f"Max Out-of-Place Bus BW: {max_oop_busbw['oop_busbw_gbps']:.2f} GB/s (size: {max_oop_busbw['size_bytes']} bytes)")
            print(f"Max In-Place Bus BW: {max_ip_busbw['ip_busbw_gbps']:.2f} GB/s (size: {max_ip_busbw['size_bytes']} bytes)")

            # Show size range
            sizes = [row['size_bytes'] for row in perf_data]
            print(f"Test size range: {min(sizes)} - {max(sizes)} bytes")
        else:
            print("\nNo performance data found")

    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Print summary of NCCL test log file')
    parser.add_argument('log_file', help='Path to NCCL test log file')

    args = parser.parse_args()

    if not Path(args.log_file).exists():
        print(f"Error: Log file '{args.log_file}' not found", file=sys.stderr)
        sys.exit(1)

    print_summary(args.log_file)

if __name__ == '__main__':
    main()