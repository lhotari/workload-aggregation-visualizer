#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.5.0",
#     "numpy>=1.20.0",
# ]
# ///
# for running this Python script, install uv, https://docs.astral.sh/uv/getting-started/installation/
#
# Copyright (c) 2024 Lari Hotari
# Licensed under MIT License with attribution requirements. See LICENSE file for details.
#
# insipired by Andrew Warfield's talk at FAST '23 - Building and Operating a Pretty Big Storage System (My Adventures in Amazon S3)
# especially the part "Individual workloads are bursty" starting at https://www.youtube.com/watch?v=sc3J4McebHE&t=1333s 
# and the "Effect of aggregating decorrelated workloads on net system load"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import argparse
import os
from matplotlib.backend_bases import KeyEvent

# Global variables
current_k = 3  # Grid side length (k×k grid)
current_fig = None
individual_workload_max = 1000000  # Default max value for individual workloads
aggregate_workload_max = None  # Will be calculated based on individual_workload_max and initial k if not provided
y_axis_label = "msgs/s"  # Default y-axis label
current_mode = "workload"  # Current visualization mode: "workload" or "overprovisioning"
current_k_range = None  # Current k range for overprovisioning mode

# Change from n-based cache to individual workload-based cache
workload_cache = {}  # Dictionary with workload_number as key and individual workload as value

def apply_common_styling(ax, title=None, xlabel=None, ylabel=None, grid=True, text_color='black'):
    """Apply common styling elements to an axis"""
    if title:
        ax.set_title(title, fontsize=14, weight='bold', color=text_color)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color=text_color)
    
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color=text_color, labelpad=10)
    
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve tick visibility
    ax.tick_params(axis='y', colors=text_color, labelsize=10, length=6, width=1, pad=8) 
    ax.tick_params(axis='x', colors=text_color, labelsize=10, length=6, width=1, pad=8)
    
    # Set border color
    for spine in ax.spines.values():
        spine.set_color(text_color)
        spine.set_linewidth(1.5)
    
    return ax

def draw_scale_and_labels(ax, max_value, y_label, x_label="time", use_dark_mode=True):
    """Helper function to properly draw scale and labels on an axis"""
    # Calculate tick values
    tick_values = np.linspace(0, max_value, 6)
    ax.set_yticks(tick_values)
    
    # Format tick labels
    y_labels = []
    for val in tick_values:
        if val < 1000:
            y_labels.append(f"{int(val)}")
        elif val < 1000000:
            y_labels.append(f"{int(val/1000)}K")
        else:
            y_labels.append(f"{val/1000000:.1f}M")
    
    # Set colors based on background
    text_color = 'white' if use_dark_mode else 'black'
    
    # Set labels
    ax.set_yticklabels(y_labels, color=text_color, fontsize=10)
    
    # Set the x and y axis labels
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold', color=text_color)
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold', color=text_color, labelpad=10)
    
    # Improve tick visibility
    ax.tick_params(axis='y', colors=text_color, labelsize=10, length=6, width=1, pad=8) 
    ax.tick_params(axis='x', colors=text_color, labelsize=10, length=6, width=1, pad=8)
    
    # Set border color
    border_color = text_color
    for spine in ax.spines.values():
        spine.set_color(border_color)
        spine.set_linewidth(1.5)
        
    return ax

def generate_spiky_workload(length=100, base_level=10, max_spike_height=100, spike_probability=0.03):
    """Generate a random workload with spikes"""
    # Create base workload with some noise
    workload = np.random.normal(base_level, base_level/10, length)
    
    # Ensure all values are at least positive
    workload = np.maximum(workload, 1)
    
    # Add random spikes based on probability
    has_spike = False
    for i in range(length):
        if random.random() < spike_probability:
            # Spikes can go up to 80% of the max height (reduced as requested)
            spike_height = random.uniform(max_spike_height/2, max_spike_height * 0.8)
            spike_width = random.randint(1, 3)
            
            # Create spike shape
            for j in range(max(0, i-spike_width), min(length, i+spike_width+1)):
                distance = abs(j - i)
                if distance == 0:
                    workload[j] = spike_height
                else:
                    # Decay based on distance from spike center
                    workload[j] = max(workload[j], spike_height * (1 - distance/spike_width))
            
            has_spike = True
    
    # Ensure at least one spike exists in the workload
    if not has_spike:
        # Add a guaranteed spike at a random position
        spike_pos = random.randint(0, length-1)
        spike_height = random.uniform(max_spike_height/2, max_spike_height * 0.8)
        spike_width = random.randint(1, 3)
        
        # Create spike shape
        for j in range(max(0, spike_pos-spike_width), min(length, spike_pos+spike_width+1)):
            distance = abs(j - spike_pos)
            if distance == 0:
                workload[j] = spike_height
            else:
                # Decay based on distance from spike center
                workload[j] = max(workload[j], spike_height * (1 - distance/spike_width))
    
    # Final check to ensure no values are zero or negative
    workload = np.maximum(workload, 1)
                
    return workload

def get_or_generate_workload(workload_number):
    """Get a workload from cache or generate a new one"""
    global workload_cache, individual_workload_max
    
    if workload_number in workload_cache:
        return workload_cache[workload_number]
    
    # Generate a random max height for this workload using normal distribution
    mean = individual_workload_max / 2
    std_dev = individual_workload_max / 4
    max_height = np.random.normal(mean, std_dev)
    
    # Ensure max_height is positive and doesn't exceed individual_workload_max
    max_height = max(individual_workload_max * 0.05, min(max_height, individual_workload_max))
    
    # Base level is 10% of this workload's max, with a minimum value
    base_level = max(individual_workload_max * 0.01, max_height * 0.1)
    
    # Generate the workload
    workload = generate_spiky_workload(
        length=100,
        base_level=base_level,
        max_spike_height=max_height,
        spike_probability=0.03
    )
    
    # Store in cache
    workload_cache[workload_number] = workload
    return workload

def generate_workloads(k):
    """
    Generate k² workloads without plotting
    
    Parameters:
    - k: grid side length
    
    Returns:
    - tuple of (all_workloads, aggregate)
    """
    # Calculate total number of workloads (n = k²)
    n = k * k
    
    # Initialize list for workloads
    all_workloads = []
    
    # Generate or retrieve individual workloads
    for i in range(n):
        workload = get_or_generate_workload(i)
        all_workloads.append(workload)
    
    # Calculate aggregate workload
    aggregate = np.sum(all_workloads, axis=0)
    
    return all_workloads, aggregate

def create_or_reuse_figure(figsize=(16, 9), clear=True):
    """Centralized function to create a new figure or reuse the existing one"""
    global current_fig
    
    if current_fig is None or not plt.fignum_exists(current_fig.number):
        current_fig = plt.figure(figsize=figsize, facecolor='white')
    else:
        plt.figure(current_fig.number)
        if clear:
            current_fig.clear()
    
    return current_fig

def plot_workload_aggregation(k=3, figsize=(16, 9), show_title=True, use_cached=True):
    """
    Plot K^2 individual workloads and their aggregate
    
    Parameters:
    - k: grid side length (k×k grid)
    - figsize: figure size in inches (default 16:9 aspect ratio)
    - show_title: whether to show the main title
    - use_cached: whether to use cached workloads if available
    
    Returns:
    - fig: The figure object
    - all_workloads: List of individual workloads
    - aggregate: The aggregated workload
    """
    global current_fig, individual_workload_max, aggregate_workload_max, y_axis_label, workload_cache
    
    # Calculate total number of workloads
    n = k * k
    
    # If not using cached data, clear the relevant keys
    if not use_cached:
        # Clear only the workloads for this specific k value
        for i in range(n):
            if i in workload_cache:
                del workload_cache[i]
    
    # Generate or retrieve workloads
    all_workloads, aggregate = generate_workloads(k)
    
    # Create or reuse figure
    fig = create_or_reuse_figure(figsize=figsize)
    
    # Create layout for individual workloads and right panel with aligned bottoms
    # Define common bottom position to align the plots
    common_bottom = 0.15
    gs_left = gridspec.GridSpec(k, k, left=0.05, right=0.45, top=0.9, bottom=common_bottom, wspace=0.1, hspace=0.1, figure=fig)
    gs_right = gridspec.GridSpec(1, 1, left=0.58, right=0.95, top=0.9, bottom=common_bottom, figure=fig)
    
    # Plot individual workloads
    axes_left = []
    for i in range(k):
        for j in range(k):
            ax = fig.add_subplot(gs_left[i, j])
            axes_left.append(ax)
            
            idx = i * k + j
            workload = all_workloads[idx]
            
            # Verify workload is not empty by checking min and max values
            wl_min = np.min(workload)
            wl_max = np.max(workload)
            
            # Add debug info if workload appears empty
            if wl_max - wl_min < individual_workload_max * 0.01:
                print(f"Warning: Workload at position [{i},{j}] (idx={idx}) has low variation: min={wl_min}, max={wl_max}")
            
            # Plot individual workload
            ax.plot(workload, color='#4a2e83', linewidth=0.8)
            
            # Calculate average workload for this individual workload
            avg_workload = np.mean(workload)
            
            # Add horizontal line for average workload (thin grey line, no text)
            ax.axhline(y=avg_workload, color='#999999', linestyle='-', linewidth=0.8, alpha=0.7)
            
            # Use the individual_workload_max for all individual plots
            ax.set_ylim(0, individual_workload_max)
            
            # Ensure linear scale
            ax.set_yscale('linear')
            
            ax.set_xticks([])
            ax.set_yticks([])
            # Add thin frame instead of hiding spines
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#CCCCCC')  # Light gray color
                spine.set_linewidth(0.5)    # Thin line width
            ax.set_facecolor('white')
            ax.grid(True, linestyle='-', alpha=0.2)
    
    # Create right side for aggregate plot
    right_ax = fig.add_subplot(gs_right[0, 0], facecolor='white')
    
    # Plot aggregate workload
    right_ax.plot(aggregate, color='#4a2e83', linewidth=2)
    
    # Calculate aggregate average, min, and max
    agg_avg = np.mean(aggregate)
    min_val = np.min(aggregate)
    max_val = np.max(aggregate)
    
    # Add horizontal lines for min, max, and average values
    right_ax.axhline(y=min_val, color='#777777', linestyle='--', linewidth=1.5, alpha=0.7)
    right_ax.axhline(y=max_val, color='#777777', linestyle='--', linewidth=1.5, alpha=0.7)
    right_ax.axhline(y=agg_avg, color='#999999', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Use the predetermined max value for aggregate
    right_ax.set_ylim(0, aggregate_workload_max)
    
    # Add proper scales and labels to the right axis
    # Remove x-axis ticks as requested
    right_ax.set_xticks([])
    
    # Calculate tick values for y-axis
    tick_values = np.linspace(0, aggregate_workload_max, 6)
    right_ax.set_yticks(tick_values)
    
    # Format y tick labels with K/M suffixes
    y_labels = []
    for val in tick_values:
        if val < 1000:
            y_labels.append(f"{int(val)}")
        elif val < 1000000:
            y_labels.append(f"{int(val/1000)}K")
        else:
            y_labels.append(f"{val/1000000:.1f}M")
    
    right_ax.set_yticklabels(y_labels)
    
    # Add labels with proper positioning
    right_ax.set_xlabel("time", fontsize=12, fontweight='bold')
    right_ax.set_ylabel(y_axis_label, fontsize=12, fontweight='bold', rotation=90, labelpad=10)
    
    # Style the ticks and grid
    right_ax.tick_params(axis='both', labelsize=10)
    right_ax.grid(True, color='#CCCCCC', linestyle='-', alpha=0.5)
    
    # Set main title if requested - with better spacing
    if show_title:
        fig.suptitle(f'Effect of aggregating decorrelated workloads on net system load (n={n})', 
                     fontsize=20, weight='bold', y=0.99)
    
    # Add titles for both sides with better positioning
    # Moved lower to prevent overlap with main title
    fig.text(0.25, 0.92, 'Individual workloads', fontsize=14, weight='bold', ha='center')
    fig.text(0.76, 0.92, 'Aggregate', fontsize=14, weight='bold', ha='center')
    
    # Force a draw to make sure labels are rendered correctly
    fig.canvas.draw()
    
    return fig, all_workloads, aggregate

def calculate_overprovisioning_factors(k_range):
    """
    Calculate overprovisioning factors (Max/Avg ratio) for different k values
    using cached workloads when available
    
    Parameters:
    - k_range: list of k values to calculate overprovisioning factors for
    
    Returns:
    - Dictionary with k values as keys and tuples of (factor, avg_val, max_val, n) as values
      where n is the number of workloads (k²)
    """
    factors = {}
    
    for k in k_range:
        print(f"Calculating overprovisioning factor for k={k}...")
        
        # Get workloads from cache or generate new ones
        all_workloads, aggregate = generate_workloads(k)
        
        # Calculate number of workloads
        n = k * k
        
        # Find avg and max values of the aggregate
        avg_val = np.mean(aggregate)
        max_val = np.max(aggregate)
        
        # Calculate overprovisioning factor (Max/Avg ratio)
        factor = max_val / avg_val if avg_val > 0 else 0
        
        factors[k] = (factor, avg_val, max_val, n)
    
    return factors

def plot_overprovisioning_factor(k_range, figsize=(16, 9), show_title=True, save_path=None, dpi=300):
    """
    Plot the overprovisioning factor (Max/Avg ratio) for different k values
    
    Parameters:
    - k_range: list of k values to plot
    - figsize: figure size in inches (default 16:9 aspect ratio)
    - show_title: whether to show the title
    - save_path: path to save the plot, if None, don't save
    - dpi: DPI for saved image
    
    Returns:
    - Figure object
    """
    global current_fig
    
    print("Calculating overprovisioning factors...")
    # Calculate overprovisioning factors for each k
    factors_data = calculate_overprovisioning_factors(k_range)
    print("Calculations complete, creating plot...")
    
    # Extract data for plotting
    ks = sorted(k_range)
    num_workloads = [factors_data[k][3] for k in ks]  # n = k²
    factor_values = [factors_data[k][0] for k in ks]  # Max/Avg ratio
    
    # Create or reuse figure
    fig = create_or_reuse_figure(figsize=figsize)
    
    # Create main plot
    ax = fig.add_subplot(111)
    
    # Plot overprovisioning factor (Max/Avg ratio)
    ax.plot(num_workloads, factor_values, 'o-', color='#4a2e83', linewidth=2.5, 
            markersize=8, markerfacecolor='white', markeredgewidth=2)
    
    # Add grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of workloads (n)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overprovisioning Factor (Max/Avg)', fontsize=12, fontweight='bold')
    
    # Format x-axis to show workload numbers
    ax.set_xticks(num_workloads)
    # Only show subset of labels if there are many points
    if len(num_workloads) > 10:
        # Show approx 5-8 labels
        step = max(len(num_workloads) // 6, 1)
        visible_ticks = num_workloads[::step]
        if num_workloads[-1] not in visible_ticks:
            visible_ticks.append(num_workloads[-1])
        ax.set_xticks(visible_ticks)
    
    # Set y-axis limits with some margin
    max_factor = max(factor_values)
    ax.set_ylim(1, max_factor * 1.1)  # Add 10% margin on top
    
    # Add annotations for select points (first, last, and a few in between)
    annotation_indices = [0]  # Always include first point
    if len(num_workloads) > 2:
        # Add a couple of points in between
        middle_points = [len(num_workloads) // 3, 2 * len(num_workloads) // 3]
        annotation_indices.extend(middle_points)
    annotation_indices.append(len(num_workloads) - 1)  # Always include last point
    
    # Add annotations
    for i in annotation_indices:
        n = num_workloads[i]
        factor = factor_values[i]
        
        # Format factor to 1 decimal place
        factor_text = f"{factor:.1f}x"
        
        # Add text annotation above the point
        ax.annotate(factor_text, 
                   xy=(n, factor), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=10,
                   weight='bold')
    
    # Add explanatory annotations for first and last point
    if len(num_workloads) > 0:
        first_n = num_workloads[0]
        first_factor = factor_values[0]
        last_n = num_workloads[-1]
        last_factor = factor_values[-1]
        
        # First point annotation (high overprovisioning)
        ax.annotate(f"At n={first_n}, system needs {first_factor:.1f}x peak capacity compared to average", 
                   xy=(first_n, first_factor),
                   xytext=(first_n + 5, first_factor + (max_factor - 1) * 0.3),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#555555'),
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Last point annotation (lower overprovisioning)
        ax.annotate(f"At n={last_n}, system needs {last_factor:.1f}x peek capacity compared to average", 
                   xy=(last_n, last_factor),
                   xytext=(last_n * 0.7, last_factor + (max_factor - 1) * 0.5),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-.2", color='#555555'),
                   fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
   
    # Set title if requested
    if show_title:
        fig.suptitle('Overprovisioning Factor Decreases with Workload Aggregation', 
                     fontsize=16, weight='bold', y=0.98)
        # Add explanatory text box about overprovisioning factor
        explanation_text = "Lower values indicate more efficient resource utilization. As workloads increase, systems require less overprovisioning capacity relative to average demand."
        fig.text(0.5, 0.01, explanation_text, fontsize=10, 
             horizontalalignment='center', fontweight='normal', 
             style='italic', wrap=True)
    
    # Tight layout with more bottom padding
    plt.tight_layout(rect=[0, 0.01, 1, 0.95])
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved overprovisioning factor plot to {save_path}")
    
    # Make sure figure gets drawn
    fig.canvas.draw()
    print("Plot created successfully")
    
    return fig
   
def generate_diagrams(k_values, output_format='png', output_dir='.', show_title=True, show_plot=True, dpi=300):
    """
    Generate diagrams for multiple grid sizes
    
    Parameters:
    - k_values: list of grid side lengths to generate
    - output_format: 'png' or 'svg' or comma-separated list 'png,svg'
    - output_dir: directory to save the files
    - show_title: whether to show the main title
    - show_plot: whether to display the plot using plt.show()
    - dpi: DPI value for exported images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process comma-separated format list if provided
    formats = [f.strip() for f in output_format.split(',')]
    
    for k in k_values:
        n = k * k  # Total number of workloads
        print(f"Generating {k}x{k} grid diagram ({n} workloads)...")
        fig, _, _ = plot_workload_aggregation(k=k, show_title=show_title)
        
        # Save the figure in each specified format
        for fmt in formats:
            filename = os.path.join(output_dir, f"aggregated_workloads_{k}x{k}.{fmt}")
            fig.savefig(filename, dpi=dpi, format=fmt)
            print(f"Saved to {filename} (at {dpi} DPI)")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

def generate_overprovisioning_diagram(k_range, output_format='png', output_dir='.', show_title=True, show_plot=True, dpi=300):
    """
    Generate an overprovisioning factor diagram showing Max/Avg ratios for different k values
    
    Parameters:
    - k_range: list of k values to include in the diagram
    - output_format: 'png' or 'svg' or comma-separated list 'png,svg'
    - output_dir: directory to save the file
    - show_title: whether to show the main title
    - show_plot: whether to display the plot using plt.show()
    - dpi: DPI value for exported images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating overprovisioning factor diagram for k range {min(k_range)} to {max(k_range)}...")
    
    # Process comma-separated format list if provided
    formats = [f.strip() for f in output_format.split(',')]
    
    # Create the overprovisioning diagram
    fig = plot_overprovisioning_factor(k_range, show_title=show_title)
    
    # Save in each specified format
    for fmt in formats:
        filename = os.path.join(output_dir, f"overprovisioning_factor_k{min(k_range)}-{max(k_range)}.{fmt}")
        fig.savefig(filename, dpi=dpi, format=fmt)
        print(f"Saved to {filename} (at {dpi} DPI)")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def switch_visualization_mode(new_mode):
    """
    Switch between visualization modes while preserving workload data
    
    Parameters:
    - new_mode: the mode to switch to ('workload' or 'overprovisioning')
    """
    global current_mode, current_k_range, workload_cache
    
    if new_mode == current_mode:
        # No change needed
        return
    
    print(f"Switching from {current_mode} mode to {new_mode} mode...")
    
    if new_mode == "workload":
        # Switch from overprovisioning to workload
        plot_workload_aggregation(k=current_k, use_cached=True)
        print(f"Switched to workload view with k={current_k} (n={current_k*current_k})")
    elif new_mode == "overprovisioning":
        # Switch from workload to overprovisioning
        
        # Determine k values based on cached workloads
        cached_workload_indices = sorted(list(workload_cache.keys()))
        
        if not cached_workload_indices:
            # If no cache exists yet, make sure the current_k is cached
            generate_workloads(current_k)
            cached_k_values = [current_k]
        else:
            # Determine unique k values that would give us these workload numbers
            # We need to find the largest k where k² ≤ max_index+1
            max_index = max(cached_workload_indices) if cached_workload_indices else 0
            # Find the largest perfect square ≤ max_index+1
            largest_k = int(np.sqrt(max_index + 1))
            # Generate a list of k values (1 to largest_k)
            cached_k_values = list(range(1, largest_k + 1))
        
        # Always ensure current_k is in the list
        if current_k not in cached_k_values:
            generate_workloads(current_k)
            cached_k_values.append(current_k)
            cached_k_values.sort()
        
        # Use cached values for the range
        current_k_range = cached_k_values
        
        plot_overprovisioning_factor(current_k_range)
        k_values_str = ', '.join(str(k) for k in cached_k_values)
        print(f"Switched to overprovisioning view using {len(current_k_range)} cached k values: {k_values_str}")
    
    current_mode = new_mode

def print_common_controls():
    """Print common controls for both modes"""
    print("Controls:")
    print("  Tab                    : Switch between workload and overprovisioning modes")
    print("  Space/Enter            : Regenerate random chart")
    print("  s                      : Save current diagram")
    print("  ESC                    : Exit")

def interactive_workload_mode(initial_k=3, show_title=True):
    """
    Start interactive mode for workload visualization
    """
    global current_k, current_fig, current_mode
    current_k = initial_k
    current_mode = "workload"
    
    # Create initial plot
    fig, _, _ = plot_workload_aggregation(k=current_k, show_title=show_title)
    
    # Set window title
    fig.canvas.manager.set_window_title("Effect of aggregating decorrelated workloads on net system load")
    
    # Connect the key press event handler
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, initial_k))
    
    n = current_k * current_k
    print(f"Interactive mode: Grid size {current_k}x{current_k} ({n} workloads)")
    
    print_common_controls()
    print("Additional controls:")
    print("  Left/Right arrows, +/- : Change grid size")
    print("  Home                   : Reset to initial grid size")
    print("  Numbers 1-9            : Set specific grid size")
    print("  0                      : Reset to 1x1 grid")
    
    plt.show()

def interactive_overprovisioning_mode(k_range, show_title=True):
    """
    Start interactive mode for overprovisioning factor visualization
    
    Parameters:
    - k_range: list of k values to plot
    - show_title: whether to show the title
    """
    global current_fig, current_mode, current_k_range
    current_mode = "overprovisioning"
    current_k_range = k_range
    
    print("Starting interactive overprovisioning mode...")

    # Make sure all k values in the range are cached
    for k in k_range:
        if k not in workload_cache:
            generate_workloads(k)
    
    # Create initial plot with progress feedback
    fig = plot_overprovisioning_factor(k_range, show_title=show_title)
    
    # Set window title
    fig.canvas.manager.set_window_title("Overprovisioning Factor Visualization")
    
    # Connect the key press event handler
    fig.canvas.mpl_connect('key_press_event', lambda event: on_key_press(event, k_range=k_range))
    
    print(f"Interactive overprovisioning mode for {len(k_range)} k values")
    print("Controls:")
    print("  Tab                    : Switch between overprovisioning and workload modes")
    print("  Space/Enter            : Regenerate random chart")
    print("  Right arrow/+/=        : Add next higher k value")
    print("  Left arrow/-           : Remove highest k value")
    print("  Up arrow/Page Up       : Remove lowest k value")
    print("  Down arrow/Page Down   : Add next lower k value (if k>1)")
    print("  Numbers 1-9            : Set to use only that specific k value")
    print("  0                      : Reset to standard k range 1-10")
    print("  c                      : Add current_k value to the analysis")
    print("  a                      : Use all cached k values")
    print("  s                      : Save current diagram")
    print("  ESC                    : Exit")
    
    plt.show()

def get_output_filename(prefix, k_value=None, k_range=None, timestamp=None, output_format='png'):
    """Create standardized output filename"""
    if timestamp is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create base filename without extension if output_format is empty
    if not output_format:
        extension = ""
    else:
        extension = f".{output_format}"
    
    if k_value is not None:
        return f"{prefix}_{k_value}x{k_value}_{timestamp}{extension}"
    
    if k_range is not None:
        if len(k_range) <= 5:
            # For a small number of values, list them all
            k_str = '_'.join(str(k) for k in k_range)
        else:
            # For many values, just show range
            k_str = f"{min(k_range)}-{max(k_range)}"
        return f"{prefix}_{k_str}_{timestamp}{extension}"
    
    return f"{prefix}_{timestamp}{extension}"

def handle_common_keys(event, dpi=300):
    """Handle key presses common to all visualization modes"""
    global current_mode, workload_cache, current_k, current_k_range
    
    if event.key in ['enter', ' ']:  # enter or space
        # Regenerate the current chart
        if current_mode == "workload":
            # Clear cache for current k workloads only
            n = current_k * current_k
            for i in range(n):
                if i in workload_cache:
                    del workload_cache[i]
            plot_workload_aggregation(k=current_k, use_cached=False)  # Force regeneration
            print(f"Regenerated workload chart with grid size {current_k}x{current_k} ({n} workloads)")
        elif current_mode == "overprovisioning" and current_k_range:
            # Clear cache for all workloads corresponding to k values in range
            for k in current_k_range:
                n = k * k
                for i in range(n):
                    if i in workload_cache:
                        del workload_cache[i]
            plot_overprovisioning_factor(current_k_range)
            print(f"Regenerated overprovisioning factor chart for k range {min(current_k_range)} to {max(current_k_range)}")
        return True
    elif event.key == 's':
        # Save the current figure
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Default to png format for interactive mode
        formats = ['png']
        
        if current_mode == "workload":
            base_filename = get_output_filename("aggregated_workloads", k_value=current_k, timestamp=timestamp, output_format="")
        else:
            base_filename = get_output_filename("overprovisioning_factor", k_range=current_k_range, timestamp=timestamp, output_format="")
        
        # Save in each format
        for fmt in formats:
            filename = f"{base_filename}{fmt}"
            plt.gcf().savefig(filename, dpi=dpi, format=fmt)
            print(f"Saved to {filename} (at {dpi} DPI)")
        return True
    elif event.key == 'escape':
        # Exit the application
        print("Exiting...")
        plt.close()
        import sys
        sys.exit(0)
        return True
    elif event.key == 'tab':
        # Switch between modes
        new_mode = "overprovisioning" if current_mode == "workload" else "workload"
        switch_visualization_mode(new_mode)
        return True
        
    return False  # Key not handled

def on_key_press(event, initial_k=3, k_range=None, dpi=300):
    """
    Unified key press event handler for both visualization modes
    
    Parameters:
    - event: Key press event
    - initial_k: Initial k value for workload mode
    - k_range: k range for overprovisioning mode
    - dpi: DPI for saved images
    """
    global current_k, current_mode, current_k_range, workload_cache
    
    # Print key press for debugging (can be commented out in production)
    #print(f"Key pressed: '{event.key}'")
    
    # First try to handle common keys
    if handle_common_keys(event, dpi):
        plt.draw()
        return
    
    # Now handle mode-specific keys
    if current_mode == "workload":
        if event.key in ['right', '+', '=', 'pagedown']:  # '=' is on the same key as '+' without shift
            # Increase k and redraw
            current_k += 1
            plot_workload_aggregation(k=current_k)
            n = current_k * current_k
            print(f"Grid size increased to {current_k}x{current_k} ({n} workloads)")
            plt.draw()
        elif event.key in ['left', '-', 'pageup'] and current_k > 1:
            # Decrease k and redraw, but not below 1
            current_k -= 1
            plot_workload_aggregation(k=current_k)
            n = current_k * current_k
            print(f"Grid size decreased to {current_k}x{current_k} ({n} workloads)")
            plt.draw()
        # Support multiple possible names for the Home key
        elif event.key in ['home', 'Home', 'begin', 'pos1', 'cmd+left']:
            # Reset to initial k value
            current_k = initial_k
            plot_workload_aggregation(k=current_k)
            n = current_k * current_k
            print(f"Grid size reset to initial {current_k}x{current_k} ({n} workloads)")
            plt.draw()
        elif event.key == '0':
            # Reset to k=1
            current_k = 1
            plot_workload_aggregation(k=current_k)
            n = current_k * current_k
            print(f"Grid size reset to {current_k}x{current_k} ({n} workloads)")
            plt.draw()
        elif event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            # Set k directly to the pressed number
            new_k = int(event.key)
            current_k = new_k
            plot_workload_aggregation(k=current_k)
            n = current_k * current_k
            print(f"Grid size set to {current_k}x{current_k} ({n} workloads)")
            plt.draw()
    elif current_mode == "overprovisioning" and current_k_range:
        # Get the current range values
        min_k = min(current_k_range)
        max_k = max(current_k_range)
        
        if event.key in ['right', '+', '=']:  # Increase max k
            new_max_k = max_k + 1
            # Add the new k value to the range and ensure it's in the cache
            generate_workloads(new_max_k)
            current_k_range.append(new_max_k)
            current_k_range.sort()
            plot_overprovisioning_factor(current_k_range)
            plt.draw()
            print(f"Added k={new_max_k} to overprovisioning analysis")
        elif event.key in ['left', '-'] and len(current_k_range) > 1:  # Remove largest k
            new_range = [k for k in current_k_range if k != max_k]
            if new_range:  # Ensure we don't end up with an empty range
                current_k_range = new_range
                plot_overprovisioning_factor(new_range)
                plt.draw()
                print(f"Removed k={max_k} from overprovisioning analysis")
        elif event.key in ['up', 'pageup'] and len(current_k_range) > 1:  # Remove smallest k
            new_range = [k for k in current_k_range if k != min_k]
            if new_range:  # Ensure we don't end up with an empty range
                current_k_range = new_range
                plot_overprovisioning_factor(new_range)
                plt.draw()
                print(f"Removed k={min_k} from overprovisioning analysis")
        elif event.key in ['down', 'pagedown']:  # Add a smaller k if possible
            if min_k > 1:
                new_min_k = min_k - 1
                # Add the new k value to the range and ensure it's in the cache
                generate_workloads(new_min_k)
                current_k_range.append(new_min_k)
                current_k_range.sort()
                plot_overprovisioning_factor(current_k_range)
                plt.draw()
                print(f"Added k={new_min_k} to overprovisioning analysis")
        elif event.key == '0':  # Reset to standard range (1-10)
            new_range = list(range(1, 11))
            # Ensure all k values are in the cache
            for k in new_range:
                if k not in current_k_range:
                    generate_workloads(k)
            current_k_range = new_range
            plot_overprovisioning_factor(new_range)
            plt.draw()
            print(f"Reset to standard k range 1-10")
        elif event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            # Set range to use only this k value
            new_k = int(event.key)
            generate_workloads(new_k)
            current_k_range = [new_k]
            plot_overprovisioning_factor(current_k_range)
            plt.draw()
            print(f"Set overprovisioning analysis to use only k={new_k}")
        elif event.key == 'c':  # Custom key to add current_k to the range
            if current_k not in current_k_range:
                current_k_range.append(current_k)
                current_k_range.sort()
                plot_overprovisioning_factor(current_k_range)
                plt.draw()
                print(f"Added current k={current_k} to overprovisioning analysis")
        elif event.key == 'a':  # Add all cached k values to the range
            # Find unique k values from workload cache
            cached_workload_indices = sorted(list(workload_cache.keys()))
            max_index = max(cached_workload_indices) if cached_workload_indices else 0
            largest_k = int(np.sqrt(max_index + 1))
            all_k_values = list(range(1, largest_k + 1))
            
            current_k_range = all_k_values
            plot_overprovisioning_factor(current_k_range)
            plt.draw()
            print("Using all cached k values for overprovisioning analysis")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Generate workload aggregation visualization.')
    parser.add_argument('-k', '--grid-size', type=int, default=3,
                        help='Grid side length (k×k grid, default: 3)')
    parser.add_argument('--no-title', action='store_true',
                        help='Disable the main title')
    parser.add_argument('--format', type=str, default='png',
                        help='Output file format(s) (comma-separated, e.g., png,svg) (default: png)')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for saved files (default: current directory)')
    parser.add_argument('--range', type=str, 
                        help='Generate multiple diagrams with grid sizes in range START-END, e.g., 3-9')
    parser.add_argument('--batch', action='store_true',
                        help='Generate diagrams without showing the UI')
    # Interactive mode is now default, so this argument is removed
    parser.add_argument('--individual-max', type=float, default=1000000,
                        help='Maximum value for individual workload y-axis (default: 1000000)')
    parser.add_argument('--aggregate-max', type=float,
                        help='Maximum value for aggregate workload y-axis (default: k*k*individual-max)')
    parser.add_argument('--y-axis-label', type=str, default='msgs/s',
                        help='Label for the y-axis (default: msgs/s)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI (dots per inch) for saved images (default: 300)')
    parser.add_argument('--plot-overprovisioning', action='store_true',
                        help='Generate a plot showing overprovisioning factor (Max/Min) vs. number of workloads')
    parser.add_argument('--overprovisioning-range', type=str, default='1-10',
                        help='Range of k values for overprovisioning factor plot (default: 1-10)')
    
    args = parser.parse_args()
    
    # Configure matplotlib - always hide toolbar by default
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['toolbar'] = 'None'
    
    # Validate that k is at least 1
    if args.grid_size < 1:
        parser.error("Grid size must be at least 1")
    
    # Set global variables from command line args
    individual_workload_max = args.individual_max
    y_axis_label = args.y_axis_label
    
    # Calculate aggregate_workload_max if not provided
    if args.aggregate_max is not None:
        aggregate_workload_max = args.aggregate_max
    else:
        # Calculate based on individual_workload_max and initial k
        initial_k = args.grid_size
        aggregate_workload_max = individual_workload_max * initial_k * initial_k
    
    # Process overprovisioning factor plot if requested
    if args.plot_overprovisioning:
        try:
            start, end = map(int, args.overprovisioning_range.split('-'))
            # Validate range values
            if start < 1 or end < 1:
                parser.error("Grid sizes in range must be at least 1")
            if start > end:
                parser.error("Invalid range: START must be less than or equal to END")
                
            k_range = list(range(start, end + 1))
            
            if args.batch:
                # Generate static overprovisioning factor diagram with no UI in batch mode
                generate_overprovisioning_diagram(
                    k_range=k_range,
                    output_format=args.format,
                    output_dir=args.output_dir,
                    show_title=not args.no_title,
                    show_plot=False,
                    dpi=args.dpi
                )
                
                # New functionality: Generate individual workload diagrams for each k value
                # using the already cached data from the overprovisioning calculation
                print(f"\nGenerating individual workload diagrams for each k value in range {start}-{end}...")
                generate_diagrams(
                    k_values=k_range,
                    output_format=args.format,
                    output_dir=args.output_dir,
                    show_title=not args.no_title,
                    show_plot=False,
                    dpi=args.dpi
                )
                print(f"Completed generating all diagrams for k values {start}-{end}")
            else:
                # Interactive overprovisioning mode is the default
                interactive_overprovisioning_mode(k_range, show_title=not args.no_title)
            
            # Exit after overprovisioning plot handling is complete
            import sys
            sys.exit(0)
                
        except ValueError:
            parser.error(f"Invalid range format '{args.overprovisioning_range}'. Use format START-END (e.g., 1-25)")
    
    # Process based on the chosen mode for regular visualizations
    # Handle standard workload visualization mode
    
    # Process range if specified
    if args.range:
        try:
            start, end = map(int, args.range.split('-'))
            # Validate range values
            if start < 1 or end < 1:
                parser.error("Grid sizes in range must be at least 1")
            if start > end:
                parser.error("Invalid range: START must be less than or equal to END")
                
            k_values = list(range(start, end + 1))
            
            if args.batch:
                # Generate diagrams without UI
                generate_diagrams(
                    k_values=k_values,
                    output_format=args.format,
                    output_dir=args.output_dir,
                    show_title=not args.no_title,
                    show_plot=False,
                    dpi=args.dpi
                )
                # Exit after generating all diagrams
                import sys
                sys.exit(0)
            else:
                # We'll only generate one diagram at a time in interactive mode
                # with the range option, but start at the first k value
                print(f"Starting interactive mode with initial k={start}")
                print("Note: The range option in interactive mode only sets the starting k value")
                interactive_workload_mode(initial_k=start, show_title=not args.no_title)
        except ValueError:
            parser.error(f"Invalid range format '{args.range}'. Use format START-END (e.g., 3-9)")
    else:
        # Single k value mode
        if args.batch:
            # Just generate a single diagram in batch mode
            k = args.grid_size
            n = k * k
            fig, _, _ = plot_workload_aggregation(k=k, show_title=not args.no_title)
            
            # Process comma-separated format list
            formats = [f.strip() for f in args.format.split(',')]
            
            # Save the figure in each specified format
            os.makedirs(args.output_dir, exist_ok=True)
            for fmt in formats:
                filename = os.path.join(args.output_dir, 
                                         get_output_filename("aggregated_workloads", k_value=k, output_format=fmt))
                fig.savefig(filename, dpi=args.dpi, format=fmt)
                print(f"Saved to {filename} (at {args.dpi} DPI)")
            
            plt.close(fig)
            # Exit after saving
            import sys
            sys.exit(0)
        else:
            # Default is now interactive mode
            interactive_workload_mode(initial_k=args.grid_size, show_title=not args.no_title)