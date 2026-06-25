import argparse
import pandas as pd
from pathlib import Path

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the Darshan aggregated data from CSV or Parquet."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    print(f"Loading data from {file_path}...")
    if path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path)

def process_applications(df: pd.DataFrame, min_jobs: int = 3, min_volume_gb: float = 10.0, min_runtime_hours: float = 1.0):
    """
    Groups data by executable to identify overarching I/O patterns, 
    tracks explicit write volumes, and provides a breakdown of pattern categories.
    """
    print(f"\nApplication I/O Pattern Profiling")
    print(f"Filtering criteria: >= {min_jobs} jobs AND (>= {min_volume_gb} GB OR >= {min_runtime_hours} hours)")
    
    # 1. Filter by job count first to save computation
    job_counts = df['exe'].value_counts()
    recurring_apps = job_counts[job_counts >= min_jobs].index
    
    if len(recurring_apps) == 0:
        print(f"[CLEAR] No applications have run {min_jobs} or more times.")
        return
        
    df_apps = df[df['exe'].isin(recurring_apps)].copy()
    
    # 2. Group by application
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
        write_time=('POSIX_F_WRITE_TIME', 'sum')  # Pulled up front for better performance
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
    
    # 3. Apply Volume and Runtime Thresholds
    heavy_hitters = app_profiles[
        (app_profiles['Total Vol (GB)'] >= min_volume_gb) | 
        (app_profiles['Total Runtime (Hrs)'] >= min_runtime_hours)
    ].copy()
    
    if heavy_hitters.empty:
        print(f"[CLEAR] No applications met the heavy-hitter thresholds.")
        return
    
    # 4. Calculate Ratios for the remaining heavy hitters
    total_accesses = heavy_hitters['posix_reads'] + heavy_hitters['posix_writes']
    total_seq = heavy_hitters['posix_seq_reads'] + heavy_hitters['posix_seq_writes']
    
    heavy_hitters['read_ratio'] = (heavy_hitters['posix_read'] / (heavy_hitters['posix_read'] + heavy_hitters['posix_write'])).fillna(0)
    heavy_hitters['seq_ratio'] = (total_seq / total_accesses).fillna(0)
    heavy_hitters['uses_high_level'] = (heavy_hitters['h5_total'] > 0) | (heavy_hitters['nc_total'] > 0)
    
    # 5. Assign Pattern Tags
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
    
    display_cols = ['total_jobs', 'Total Runtime (Hrs)', 'Read Vol (GB)', 'Write Vol (GB)', 'Total Vol (GB)', 'Read %', 'Seq %', 'Pattern']
    sorted_profiles = heavy_hitters.sort_values('Total Vol (GB)', ascending=False)
    
    print(f"\nProfiled {len(heavy_hitters)} heavy-hitter applications:\n")
    print(sorted_profiles[display_cols].to_string())
    
    sorted_profiles.to_csv('report_app_profiles.csv')
    print("\n-> Full application profiles saved to report_app_profiles.csv")
    
    # 6. New Step: Generate and Display Overall Pattern Summary Table
    print("\n" + "="*85)
    print("OVERALL I/O PATTERN GROUP SUMMARY")
    print("="*85)
    
    summary_profiles = heavy_hitters.groupby('Pattern').agg(
        num_apps=('total_jobs', 'count'),
        total_jobs=('total_jobs', 'sum'),
        total_read_vol=('Read Vol (GB)', 'sum'),
        total_write_vol=('Write Vol (GB)', 'sum'),
        total_combined_vol=('Total Vol (GB)', 'sum')
    ).sort_values('total_combined_vol', ascending=False)
    
    # Print the grouped breakdown table cleanly
    headers = ['Apps Count', 'Total Jobs Run', 'Total Read (GB)', 'Total Write (GB)', 'Combined Vol (GB)']
    print(summary_profiles.to_string(header=headers))
    
    summary_profiles.to_csv('report_pattern_summary.csv')
    print("\n-> Pattern group summary saved to report_pattern_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze aggregated Darshan I/O data for HPC anti-patterns.")
    parser.add_argument("data_file", type=str, help="Path to the .csv or .parquet file generated by the parser.")
    
    # Optional flags to tune thresholds
    parser.add_argument("--tiny-io-gb", type=float, default=1.0, help="Min total GB written to flag for tiny I/O (default: 1.0)")
    parser.add_argument("--tiny-io-kb", type=float, default=4.0, help="Max average write size in KB to flag as tiny (default: 4.0)")
    parser.add_argument("--meta-sec", type=float, default=60.0, help="Min metadata time in seconds to trigger warning (default: 60.0)")
    
    args = parser.parse_args()
    
    try:
        master_df = load_data(args.data_file)
        
        print(f"Data loaded successfully. Analyzing {len(master_df)} jobs...\n")
        
        process_applications(master_df, min_jobs=3)
 
        print("\n[SUCCESS] Analysis complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
