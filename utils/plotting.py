import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

def plot_nccl_performance(df: pd.DataFrame, 
                         target_num_nodes: Optional[int] = None,
                         figsize: tuple = (12, 8)) -> plt.Figure:
    """Plot NCCL out-of-place bus bandwidth vs message size grouped by job.
    
    Args:
        df: DataFrame from performance_data_to_dataframe()
        target_num_nodes: Filter to specific number of nodes (if None, uses most common)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    
    # Filter to target number of nodes
    if target_num_nodes is None:
        target_num_nodes = df['num_nodes'].mode().iloc[0]
    
    filtered_df = df[df['num_nodes'] == target_num_nodes].copy()
    
    if filtered_df.empty:
        raise ValueError(f"No data found for {target_num_nodes} nodes")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by jobid and create labels
    colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_df['jobid'].unique())))
    
    for i, (jobid, group) in enumerate(filtered_df.groupby('jobid')):
        # Create label with NCCL version and alt_read status
        nccl_version = group['nccl_version'].iloc[0] or 'Unknown'
        uses_alt_read = group['uses_alt_read'].iloc[0]
        alt_read_str = "alt_read" if uses_alt_read else "no_alt_read"
        
        label = f"Job {jobid} (NCCL {nccl_version}, {alt_read_str})"
        
        # Sort by message size for clean lines
        group_sorted = group.sort_values('size_bytes')
        
        # Plot line with markers
        ax.plot(group_sorted['size_bytes'], 
                group_sorted['oop_busbw_gbps'],
                marker='o', 
                linewidth=2,
                markersize=4,
                color=colors[i],
                label=label)
    
    # Formatting
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Message Size (bytes)')
    ax.set_ylabel('Out-of-Place Bus Bandwidth (GB/s)')
    ax.set_title(f'NCCL Performance Comparison ({target_num_nodes} nodes, '
                f'{filtered_df["num_gpus"].iloc[0]} GPUs)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc=0)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis ticks to show common sizes
    size_ticks = [2**i for i in range(14, 34) if 2**i <= filtered_df['size_bytes'].max()]
    ax.set_xticks(size_ticks)

    def format_size(s):
        if s >= 1024**3:
            return f'{s//(1024**3)}G'
        elif s >= 1024**2:
            return f'{s//(1024**2)}M'
        elif s >= 1024:
            return f'{s//1024}K'
        else:
            return f'{s}B'

    ax.set_xticklabels([format_size(s) for s in size_ticks], rotation=45)
    
    plt.tight_layout()
    return fig

def plot_nccl_comparison_matrix(df: pd.DataFrame, 
                               metric: str = 'oop_busbw_gbps',
                               figsize: tuple = (15, 10)) -> plt.Figure:
    """Create a matrix of subplots comparing different node counts.
    
    Args:
        df: DataFrame from performance_data_to_dataframe()
        metric: Column name to plot ('oop_busbw_gbps', 'ip_busbw_gbps', etc.)
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object
    """
    
    node_counts = sorted(df['num_nodes'].unique())
    n_plots = len(node_counts)
    
    # Calculate subplot grid
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df['jobid'].unique())))
    color_map = {jobid: colors[i] for i, jobid in enumerate(df['jobid'].unique())}
    
    for plot_idx, num_nodes in enumerate(node_counts):
        ax = axes[plot_idx]
        node_df = df[df['num_nodes'] == num_nodes]
        
        for jobid, group in node_df.groupby('jobid'):
            nccl_version = group['nccl_version'].iloc[0] or 'Unknown'
            uses_alt_read = group['uses_alt_read'].iloc[0]
            alt_read_str = "alt_read" if uses_alt_read else "no_alt_read"
            
            label = f"Job {jobid} ({alt_read_str})"
            group_sorted = group.sort_values('size_bytes')
            
            ax.plot(group_sorted['size_bytes'], 
                   group_sorted[metric],
                   marker='o', 
                   linewidth=2,
                   markersize=3,
                   color=color_map[jobid],
                   label=label)
        
        ax.set_xscale('log', base=2)
        ax.set_title(f'{num_nodes} nodes ({node_df["num_gpus"].iloc[0]} GPUs)')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Message Size (bytes)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        
        if plot_idx == 0:  # Only show legend on first subplot
            ax.legend(fontsize='small')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'NCCL Performance Comparison - {metric.replace("_", " ").title()}')
    plt.tight_layout()
    return fig

# Example usage function
def create_nccl_plots(parsed_results: List[Dict]) -> List[plt.Figure]:
    """Create standard NCCL performance plots from parsed results.
    
    Args:
        parsed_results: List of dictionaries from parse_nccl_log()
        
    Returns:
        List of matplotlib Figure objects
    """
    from utils.utils import performance_data_to_dataframe
    
    df = performance_data_to_dataframe(parsed_results)
    
    figures = []
    
    # Single node count plot
    try:
        fig1 = plot_nccl_performance(df)
        figures.append(fig1)
    except ValueError as e:
        print(f"Could not create single plot: {e}")
    
    # Matrix comparison plot
    if len(df['num_nodes'].unique()) > 1:
        fig2 = plot_nccl_comparison_matrix(df)
        figures.append(fig2)
    
    return figures