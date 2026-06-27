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
        return pd.read_parquet(file_path, engine='fastparquet') 
    else:
        return pd.read_csv(file_path, engine='python')

def process_applications(df: pd.DataFrame, output_prefix: str, min_jobs: int = 3, min_volume_gb: float = 10.0, min_runtime_hours: float = 1.0):
    """
    Groups data by executable to identify overarching I/O patterns, 
    tracks explicit read/write volumes, analyses metadata overhead & operation types, 
    and exports visualisations.
    """
    print(f"\nApplication I/O Pattern Profiling")
    print(f"Filtering criteria: >= {min_jobs} jobs AND (>= {min_volume_gb} GB OR >= {min_runtime_hours} hours)")
    
    job_counts = df['exe'].value_counts()
    recurring_apps = job_counts[job_counts >= min_jobs].index
    
    if len(recurring_apps) == 0:
        print(f"[CLEAR] No applications have run {min_jobs} or more times.")
        return
        
    df_apps = df[df['exe'].isin(recurring_apps)].copy()
    
    # Group by application AND pull in the metadata operation counts
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
        write_time=('POSIX_F_WRITE_TIME', 'sum'),
        # New Metadata Operation Counts
        posix_opens=('POSIX_OPENS', 'sum'),
        posix_stats=('POSIX_STATS', 'sum'),
        posix_seeks=('POSIX_SEEKS', 'sum')
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
    
    # Calculate IO Time and Metadata metrics
    app_profiles['total_io_time'] = app_profiles['meta_time'] + app_profiles['io_time'] + app_profiles['write_time']
    app_profiles['Meta Time (Hrs)'] = (app_profiles['meta_time'] / 3600).round(2)
    app_profiles['Meta %'] = ((app_profiles['meta_time'] / app_profiles['total_io_time']) * 100).fillna(0).round(1)
    
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
    
    # Calculate per-job metadata operation averages (to easily spot spamming behavior)
    heavy_hitters['Avg Opens/Job'] = (heavy_hitters['posix_opens'] / heavy_hitters['total_jobs']).fillna(0).astype(int)
    heavy_hitters['Avg Stats/Job'] = (heavy_hitters['posix_stats'] / heavy_hitters['total_jobs']).fillna(0).astype(int)
    heavy_hitters['Avg Seeks/Job'] = (heavy_hitters['posix_seeks'] / heavy_hitters['total_jobs']).fillna(0).astype(int)
    
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
    heavy_hitters['Meta % (String)'] = heavy_hitters['Meta %'].astype(str) + '%'
    
    # --- METADATA ANALYSIS SECTION ---
    print("\n" + "="*95)
    print("OVERALL METADATA OVERHEAD ANALYSIS")
    print("="*95)
    
    total_meta_hrs = heavy_hitters['meta_time'].sum() / 3600
    total_data_io_hrs = (heavy_hitters['io_time'].sum() + heavy_hitters['write_time'].sum()) / 3600
    total_all_io_hrs = total_meta_hrs + total_data_io_hrs
    global_meta_pct = (total_meta_hrs / total_all_io_hrs * 100) if total_all_io_hrs > 0 else 0
    
    print(f"Total Dataset Metadata Time: {total_meta_hrs:.2f} Hours")
    print(f"Total Dataset Read/Write Time: {total_data_io_hrs:.2f} Hours")
    print(f"Global Metadata Overhead: {global_meta_pct:.1f}% of all I/O time")
    
    if global_meta_pct > 30:
        print("[WARNING] High overall metadata overhead detected. Consider reviewing file creation/stat patterns.")
    
    # Include the operation counts in the metadata output CSV
    meta_display_cols = [
        'total_jobs', 'Meta Time (Hrs)', 'Meta % (String)', 
        'Avg Opens/Job', 'Avg Stats/Job', 'Avg Seeks/Job', 'Pattern'
    ]
    meta_sorted_profiles = heavy_hitters.sort_values('meta_time', ascending=False)
    
    print(f"\nTop 5 Applications by Metadata Overhead:\n")
    print(meta_sorted_profiles.head(5)[meta_display_cols].to_string())
    
    meta_save_filename = f"{output_prefix}_top_metadata_apps.csv"
    meta_sorted_profiles[meta_display_cols].to_csv(meta_save_filename)
    print(f"\n-> Full metadata profile saved to {meta_save_filename}")
    
    # --- STANDARD APP PROFILES SECTION ---
    print("\n" + "="*95)
    print("VOLUME & PATTERN SUMMARY")
    print("="*95)
    
    display_cols = ['total_jobs', 'Total Runtime (Hrs)', 'Read Vol (GB)', 'Write Vol (GB)', 'Total Vol (GB)', 'Read %', 'Write %', 'Seq %', 'Pattern']
    sorted_profiles = heavy_hitters.sort_values('Total Vol (GB)', ascending=False)
    
    print(f"\nProfiled {len(heavy_hitters)} heavy-hitter applications by volume:\n")
    print(sorted_profiles.head(5)[display_cols].to_string())
    print("...")
    
    save_filename = f"{output_prefix}_app_profiles.csv"
    sorted_profiles.to_csv(save_filename)
    print("\n-> Full application profiles saved to " + save_filename)
    
    # Generate and Display Overall Pattern Summary Table
    summary_profiles = heavy_hitters.groupby('Pattern').agg(
        num_apps=('total_jobs', 'count'),
        total_jobs=('total_jobs', 'sum'),
        total_read_vol=('Read Vol (GB)', 'sum'),
        total_write_vol=('Write Vol (GB)', 'sum'),
        total_combined_vol=('Total Vol (GB)', 'sum')
    ).sort_values('total_combined_vol', ascending=False)
    
    print("\nPattern Group Summary:")
    headers = ['Apps Count', 'Total Jobs Run', 'Total Read (GB)', 'Total Write (GB)', 'Combined Vol (GB)']
    print(summary_profiles.to_string(header=headers))
   
    save_filename = f"{output_prefix}_pattern_summary.csv"
    summary_profiles.to_csv(save_filename)
    print(f"\n-> Pattern group summary saved to {save_filename}")

    # --- DATA VISUALISATIONS ---
    print("\nGenerating visual reports...")
    top_n = min(10, len(sorted_profiles))
    
    # 1. Plot Top Apps by Volume Stacked Bar Chart
    top_apps = sorted_profiles.head(top_n)
    plt.figure(figsize=(12, 7))
    top_apps[['Read Vol (GB)', 'Write Vol (GB)']].plot(
        kind='bar', 
        stacked=True, 
        figsize=(12, 7),
        color=['#1f77b4', '#d62728']
    )
    plt.title(f'Top {top_n} Applications by Total I/O Volume')
    plt.ylabel('Data Volume (GB)')
    plt.xlabel('Application Executable')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_top_apps_volume.png", dpi=300)
    plt.close()
    
    # 2. Plot Overarching I/O Pattern Distribution (Pie Chart)
    plt.figure(figsize=(10, 8))
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
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_pattern_summary.png", dpi=300)
    plt.close()
    
    # 3. Plot Top Apps by Metadata Time
    top_meta = meta_sorted_profiles.head(top_n)
    plt.figure(figsize=(12, 7))
    top_meta['Meta Time (Hrs)'].plot(
        kind='bar', 
        figsize=(12, 7),
        color='#ff7f0e'
    )
    plt.title(f'Top {top_n} Applications by Metadata Time Overhead')
    plt.ylabel('Metadata Time (Hours)')
    plt.xlabel('Application Executable')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_top_apps_metadata.png", dpi=300)
    plt.close()

    # 4. Global Metadata vs IO Time (Pie Chart)
    plt.figure(figsize=(8, 8))
    meta_vs_io_data = pd.Series({
        'Metadata Time': total_meta_hrs,
        'Read/Write Data Time': total_data_io_hrs
    })
    meta_vs_io_data.plot(
        kind='pie',
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff7f0e', '#2ca02c'],
        textprops={'fontsize': 12}
    )
    plt.title('Global I/O Time Distribution: Metadata vs Data')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_global_metadata_ratio.png", dpi=300)
    plt.close()
    
    print("-> Visualisations saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse aggregated Darshan I/O data for HPC anti-patterns.")
    parser.add_argument("data_files", type=str, nargs='+', help="Path to one or more .csv or .parquet files generated by the parser.")
    parser.add_argument("--output-prefix", type=str, default=None, help="Base name for the output files (default: stem of first file or 'combined_analysis').")
    
    parser.add_argument("--tiny-io-gb", type=float, default=1.0, help="Min total GB written to flag for tiny I/O (default: 1.0)")
    parser.add_argument("--tiny-io-kb", type=float, default=4.0, help="Max average write size in KB to flag as tiny (default: 4.0)")
    parser.add_argument("--meta-sec", type=float, default=60.0, help="Min metadata time in seconds to trigger warning (default: 60.0)")
    
    args = parser.parse_args()
    
    try:
        dataframes = []
        
        for file_path in args.data_files:
            try:
                df = load_data(file_path)
                dataframes.append(df)
            except Exception as e:
                print(f"[WARNING] Could not load {file_path}: {e}")
                
        if not dataframes:
            print("[ERROR] No valid data files were loaded. Exiting.")
            exit(1)
            
        master_df = pd.concat(dataframes, ignore_index=True)
        
        print(f"All data loaded successfully. Analysing a combined total of {len(master_df)} jobs...\n")
        
        if args.output_prefix:
            out_prefix = args.output_prefix
        elif len(args.data_files) == 1:
            out_prefix = Path(args.data_files[0]).stem
        else:
            out_prefix = "combined_analysis"
            
        process_applications(master_df, out_prefix, min_jobs=3)
 
        print("\n[SUCCESS] Analysis complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
