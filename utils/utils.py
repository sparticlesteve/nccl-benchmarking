import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

def parse_nccl_log(log_file_path: str) -> Dict:
    """Parse NCCL test log file and extract performance data and metadata.

    Args:
        log_file_path: Path to the NCCL log file to parse

    Returns:
        Dict containing:
            - jobid (str|None): Job ID extracted from log
            - nccl_version (str|None): NCCL version string
            - uses_alt_read (bool): Whether FI_CXI_RDZV_PROTO=alt_read is used
            - num_nodes (int): Number of unique nodes detected
            - num_gpus (int): Total number of GPUs detected
            - avg_bus_bandwidth (float|None): Average bus bandwidth in GB/s
            - performance_data (List[Dict]|None): List of performance measurements, where each dict contains:
                - size_bytes (int): Message size in bytes
                - count_elements (int): Number of elements
                - type (str): Data type (e.g., 'float', 'int')
                - redop (str): Reduction operation
                - root (int): Root rank for operations
                - oop_time_us (float): Out-of-place operation time in microseconds
                - oop_algbw_gbps (float): Out-of-place algorithm bandwidth in GB/s
                - oop_busbw_gbps (float): Out-of-place bus bandwidth in GB/s
                - oop_wrong (int): Out-of-place error count
                - ip_time_us (float): In-place operation time in microseconds
                - ip_algbw_gbps (float): In-place algorithm bandwidth in GB/s
                - ip_busbw_gbps (float): In-place bus bandwidth in GB/s
                - ip_wrong (int): In-place error count
    """

    with open(log_file_path, 'r') as f:
        content = f.read()

    # Extract metadata
    metadata = {}

    # JobID
    jobid_match = re.search(r'JobID:\s*(\d+)', content)
    metadata['jobid'] = jobid_match.group(1) if jobid_match else None

    # NCCL Version
    nccl_version_match = re.search(r'NCCL_VERSION=([^\s\n]+)', content)
    metadata['nccl_version'] = nccl_version_match.group(1) if nccl_version_match else None

    # NCCL Algorithm
    nccl_algo_match = re.search(r'NCCL_ALGO=([^\s\n]+)', content)
    metadata['nccl_algo'] = nccl_algo_match.group(1) if nccl_algo_match else None

    # Check for FI_CXI_RDZV_PROTO=alt_read
    metadata['uses_alt_read'] = 'FI_CXI_RDZV_PROTO=alt_read' in content

    # Extract number of nodes and GPUs from device listing
    device_lines = re.findall(r'#\s+Rank\s+\d+.*?on\s+(\w+)\s+device', content)
    unique_nodes = set(device_lines)
    metadata['num_nodes'] = len(unique_nodes)
    metadata['num_gpus'] = len(device_lines)

    # Extract performance table
    table_start = content.find('#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong')
    if table_start == -1:
        metadata['performance_data'] = None
        return metadata

    # Find the end of the table (next # comment or end of relevant data)
    table_end = content.find('# Out of bounds values', table_start)
    if table_end == -1:
        table_end = len(content)

    table_section = content[table_start:table_end]

    # Parse performance data rows
    performance_data = []
    data_pattern = r'^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\w+)\s+(-?\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)'

    for line in table_section.split('\n'):
        match = re.match(data_pattern, line)
        if match:
            row = {
                'size_bytes': int(match.group(1)),
                'count_elements': int(match.group(2)),
                'type': match.group(3),
                'redop': match.group(4),
                'root': int(match.group(5)),
                'oop_time_us': float(match.group(6)),
                'oop_algbw_gbps': float(match.group(7)),
                'oop_busbw_gbps': float(match.group(8)),
                'oop_wrong': int(match.group(9)),
                'ip_time_us': float(match.group(10)),
                'ip_algbw_gbps': float(match.group(11)),
                'ip_busbw_gbps': float(match.group(12)),
                'ip_wrong': int(match.group(13))
            }
            performance_data.append(row)

    # Extract average bus bandwidth
    avg_bw_match = re.search(r'# Avg bus bandwidth\s*:\s*([0-9.]+)', content)
    metadata['avg_bus_bandwidth'] = float(avg_bw_match.group(1)) if avg_bw_match else None

    metadata['performance_data'] = performance_data
    return metadata

def parse_multiple_logs(log_directory: str) -> List[Dict]:
    """Parse multiple NCCL log files from a directory."""
    log_dir = Path(log_directory)
    results = []

    for log_file in log_dir.glob('*.out'):
        try:
            parsed_data = parse_nccl_log(str(log_file))
            parsed_data['log_file'] = str(log_file)
            results.append(parsed_data)
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

    return results

def performance_data_to_dataframe(parsed_results: List[Dict]) -> pd.DataFrame:
    """Convert parsed performance data to a pandas DataFrame."""
    all_rows = []

    for result in parsed_results:
        if result['performance_data']:
            for perf_row in result['performance_data']:
                row = {
                    'jobid': result['jobid'],
                    'nccl_version': result['nccl_version'],
                    'nccl_algo': result['nccl_algo'],
                    'uses_alt_read': result['uses_alt_read'],
                    'num_nodes': result['num_nodes'],
                    'num_gpus': result['num_gpus'],
                    'avg_bus_bandwidth': result['avg_bus_bandwidth'],
                    **perf_row
                }
                all_rows.append(row)

    return pd.DataFrame(all_rows)