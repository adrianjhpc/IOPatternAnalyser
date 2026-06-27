import argparse
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    print(f"Loading data from {file_path}...")
    if path.suffix == '.parquet':
        # Force fastparquet instead of pyarrow (requires: pip install fastparquet)
        return pd.read_parquet(file_path, engine='fastparquet') 
    else:
        # Fall back to the pure Python CSV parser. It is slower, but immune 
        # to C-level buffer overflow crashes.
        return pd.read_csv(file_path, engine='python')

def process_applications(df: pd.DataFrame, file_name, min_jobs: int = 3, min_volume_gb: float = 10.0, min_runtime_hours: float = 1.0):
    """
    Groups data by executable to identify overarching I/O patterns, 
    tracks explicit read/write volumes, and exports visualisations.
    """
    print(f"\nApplication I/O Pattern Profiling")
    print(f"Filtering criteria: >= {min_jobs} jobs AND (>= {min_volume_gb} GB OR >= {min_runtime_hours} hours)")
    
    # Filter by job count first to save computation
    job_counts = df['exe'].value_counts()
    recurring_apps = job_counts[job_counts >= min_jobs].index
    
    if len(recurring_apps) == 0:
        print(f"[CLEAR] No applications have run {min_jobs} or more times.")
        return
        
    df_apps = df[df['exe'].isin(recurring_apps)].copy()
    
    # Group by application
    app_profiles = df_apps.groupby('exe').agg(
        total_jobs=('job_id', 'count'),
        total_runtime=('runtime_sec', 'sum'),
        posix_read=('POSIX_BYTES_READ', 'sum'),
        posix_write=('POSIX_BYTES_WRITTEN', 'sum'),
        mpi_total=('MPIIO_BYTES_READ', 'sum'),
        h5_total=('H5_BYTES_READ', 'sum'),
        nc_total=('NC_BYTES_READ', 'sum'),
        posix_reads=('POSIX_READS', 'sum'),
        posix_writes=('POSIX_WRITES', 'sum'),
        posix_seq_reads=('POSIX_SEQ_READS', 'sum'),
        posix_seq_writes=('POSIX_SEQ_WRITES', 'sum'),
        meta_time=('POSIX_F_META_TIME', 'sum'),
        io_time=('POSIX_F_READ_TIME', 'sum'),
        write_time=('POSIX_F_WRITE_TIME', 'sum') 
    )
    
    # Add MPI/H5/NC writes to totals
    app_profiles['mpi_total'] += df_apps.groupby('exe')['MPIIO_BYTES_WRITTEN'].sum()
    app_profiles['h5_total'] += df_apps.groupby('exe')['H5_BYTES_WRITTEN'].sum()
    app_profiles['nc_total'] += df_apps.groupby('exe')['NC_BYTES_WRITTEN'].sum()
    
    # Calculate Breakdown Volumes (GB) and Total Runtime (Hours)
    app_profiles['Read Vol (GB)'] = (app_profiles['posix_read'] / (1024**3)).round(2)
    app_profiles['Write Vol (GB)'] = (app_profiles['posix_write'] / (1024**3)).round(2)
    
    total_rw_bytes = app_profiles['posix_read'] + app_profiles['posix_write']
    app_profiles['Total Vol (GB)'] = (total_rw_bytes / (1024**3)).round(2)
    app_profiles['Total Runtime (Hrs)'] = (app_profiles['total_runtime'] / 3600).round(2)
    
    # Apply Volume and Runtime Thresholds
    heavy_hitters = app_profiles[
        (app_profiles['Total Vol (GB)'] >= min_volume_gb) | 
        (app_profiles['Total Runtime (Hrs)'] >= min_runtime_hours)
    ].copy()
    
    if heavy_hitters.empty:
        print(f"[CLEAR] No applications met the heavy-hitter thresholds.")
        return
    
    # Calculate Ratios for the remaining heavy hitters
    total_accesses = heavy_hitters['posix_reads'] + heavy_hitters['posix_writes']
    total_seq = heavy_hitters['posix_seq_reads'] + heavy_hitters['posix_seq_writes']
    
    heavy_hitters['read_ratio'] = (heavy_hitters['posix_read'] / (heavy_hitters['posix_read'] + heavy_hitters['posix_write'])).fillna(0)
    heavy_hitters['write_ratio'] = (heavy_hitters['posix_write'] / (heavy_hitters['posix_read'] + heavy_hitters['posix_write'])).fillna(0)
    heavy_hitters['seq_ratio'] = (total_seq / total_accesses).fillna(0)
    heavy_hitters['uses_high_level'] = (heavy_hitters['h5_total'] > 0) | (heavy_hitters['nc_total'] > 0)
    
    # Assign Pattern Tags
    def assign_pattern(row):
        tags = []
        if row['uses_high_level']:
            tags.append("Structured (HDF5/NC)")
        elif row['mpi_total'] > (1024**3):
            tags.append("MPI-IO")
        else:
            tags.append("POSIX-Only")
            
        if row['read_ratio'] > 0.8:
            tags.append("Random Reader" if row['seq_ratio'] < 0.3 else "Sequential Reader")
        elif row['read_ratio'] < 0.2:
            tags.append("Sequential Writer" if row['seq_ratio'] > 0.7 else "Random Writer")
        else:
            tags.append("Mixed R/W")
            
        io_total_time = row['io_time'] + row['write_time']
        if row['meta_time'] > io_total_time and row['meta_time'] > 60:
            tags.append("Metadata-Heavy!")
            
        return " | ".join(tags)

    heavy_hitters['Pattern'] = heavy_hitters.apply(assign_pattern, axis=1)
    
    # Format for per-app display
    heavy_hitters['Seq %'] = (heavy_hitters['seq_ratio'] * 100).round(1).astype(str) + '%'
    heavy_hitters['Read %'] = (heavy_hitters['read_ratio'] * 100).round(1).astype(str) + '%'
    heavy_hitters['Write %'] = (heavy_hitters['write_ratio'] * 100).round(1).astype(str) + '%'
    
    display_cols = ['total_jobs', 'Total Runtime (Hrs)', 'Read Vol (GB)', 'Write Vol (GB)', 'Total Vol (GB)', 'Read %', 'Write %', 'Seq %', 'Pattern']
    sorted_profiles = heavy_hitters.sort_values('Total Vol (GB)', ascending=False)
    
    print(f"\nProfiled {len(heavy_hitters)} heavy-hitter applications:\n")
    print(sorted_profiles[display_cols].to_string())
    
    save_filename = file_name + '_app_profiles.csv'
    sorted_profiles.to_csv(save_filename)
    print("\n-> Full application profiles saved to " + save_filename)
    
    # Generate and Display Overall Pattern Summary Table
    print("\n" + "="*95)
    print("OVERALL I/O PATTERN GROUP SUMMARY")
    print("="*95)
    
    summary_profiles = heavy_hitters.groupby('Pattern').agg(
        num_apps=('total_jobs', 'count'),
        total_jobs=('total_jobs', 'sum'),
        total_read_vol=('Read Vol (GB)', 'sum'),
        total_write_vol=('Write Vol (GB)', 'sum'),
        total_combined_vol=('Total Vol (GB)', 'sum')
    ).sort_values('total_combined_vol', ascending=False)
    
    headers = ['Apps Count', 'Total Jobs Run', 'Total Read (GB)', 'Total Write (GB)', 'Combined Vol (GB)']
    print(summary_profiles.to_string(header=headers))
   
    save_filename = file_name + "_pattern_summary.csv"
 
    summary_profiles.to_csv(save_filename)
    print("\n-> Pattern group summary saved to report_pattern_summary.csv")

    # Generate Data Visualisations
    print("\nGenerating visual reports...")
    
    # Plot Top 10 Apps Stacked Bar Chart (Read vs Write)
    top_n = min(10, len(sorted_profiles))
    top_apps = sorted_profiles.head(top_n)
    
    plt.figure(figsize=(12, 7))
    top_apps[['Read Vol (GB)', 'Write Vol (GB)']].plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 7),
        color=['#1f77b4', '#d62728'] # Standard blue for read, red for write
    )
    plt.title(f'Top {top_n} Applications by Total I/O Volume')
    plt.ylabel('Data Volume (GB)')
    plt.xlabel('Application Executable')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_filename = file_name + "_top_apps_volume.png"
    plt.savefig(save_filename, dpi=300)
    plt.close()
    
    # Plot Overarching I/O Pattern Distribution (Pie Chart)
    plt.figure(figsize=(10, 8))
    # Filter out extremely small slices for a cleaner pie chart, group them into "Other" if necessary
    pie_data = summary_profiles['total_combined_vol']
    if len(pie_data) > 6:
        top_patterns = pie_data.nlargest(5)
        other_patterns = pd.Series([pie_data.drop(top_patterns.index).sum()], index=['Other Patterns'])
        pie_data = pd.concat([top_patterns, other_patterns])

    pie_data.plot(
        kind='pie', 
        autopct='%1.1f%%', 
        startangle=140, 
        cmap='tab20',
        textprops={'fontsize': 10}
    )
    plt.title('Total Storage Volume Breakdown by I/O Pattern')
    plt.ylabel('') # Hides the arbitrary y-axis label pandas adds to pie charts
    plt.tight_layout()
    save_filename = file_name + "_pattern_summary.png"
    plt.savefig(save_filename, dpi=300)
    plt.close()
    
    print("-> Visualisations saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse aggregated Darshan I/O data for HPC anti-patterns.")
    parser.add_argument("data_file", type=str, help="Path to the .csv or .parquet file generated by the parser.")
    
    parser.add_argument("--tiny-io-gb", type=float, default=1.0, help="Min total GB written to flag for tiny I/O (default: 1.0)")
    parser.add_argument("--tiny-io-kb", type=float, default=4.0, help="Max average write size in KB to flag as tiny (default: 4.0)")
    parser.add_argument("--meta-sec", type=float, default=60.0, help="Min metadata time in seconds to trigger warning (default: 60.0)")
    
    args = parser.parse_args()
    
    try:
        master_df = load_data(args.data_file)
        
        print(f"Data loaded successfully. Analysing {len(master_df)} jobs...\n")
        
        process_applications(master_df, args.data_file, min_jobs=3)
 
        print("\n[SUCCESS] Analysis complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
