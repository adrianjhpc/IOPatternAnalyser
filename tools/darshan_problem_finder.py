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

def analyze_tiny_io(df: pd.DataFrame, volume_threshold_gb: float = 1.0, size_threshold_kb: float = 4.0):
    """
    Finds jobs that write a massive amount of data, but in tiny increments.
    This thrashes network file systems and causes severe lock contention.
    """
    print(f"\n--- 1. Tiny I/O Offenders (< {size_threshold_kb} KB avg write) ---")
    
    # Calculate average write size, handling division by zero
    writes_mask = df['POSIX_WRITES'] > 0
    df.loc[writes_mask, 'avg_write_kb'] = (df.loc[writes_mask, 'POSIX_BYTES_WRITTEN'] / df.loc[writes_mask, 'POSIX_WRITES']) / 1024
    df['avg_write_kb'] = df['avg_write_kb'].fillna(0)

    volume_threshold_bytes = volume_threshold_gb * (1024**3)
    
    offenders = df[
        (df['POSIX_BYTES_WRITTEN'] > volume_threshold_bytes) & 
        (df['avg_write_kb'] > 0) & 
        (df['avg_write_kb'] < size_threshold_kb)
    ]
    
    if offenders.empty:
        print("[CLEAR] No tiny I/O offenders found.")
    else:
        print(f"[WARNING] Found {len(offenders)} jobs writing >{volume_threshold_gb}GB in <{size_threshold_kb}KB chunks.")
        display_cols = ['job_id', 'uid', 'exe', 'POSIX_BYTES_WRITTEN', 'POSIX_WRITES', 'avg_write_kb']
        
        # Sort by total writes to surface the worst thrashers first
        offenders = offenders.sort_values(by='POSIX_WRITES', ascending=False)
        
        # Format the bytes to GB for readability in the console
        console_df = offenders[display_cols].copy()
        console_df['POSIX_BYTES_WRITTEN'] = (console_df['POSIX_BYTES_WRITTEN'] / (1024**3)).round(2).astype(str) + ' GB'
        console_df['avg_write_kb'] = console_df['avg_write_kb'].round(2)
        
        print(console_df.head(10).to_string(index=False))
        offenders[display_cols].to_csv('report_tiny_io.csv', index=False)
        print("-> Full list saved to report_tiny_io.csv")

def analyze_metadata_hammers(df: pd.DataFrame, min_meta_time_sec: float = 60.0):
    """
    Finds jobs spending more time doing metadata operations (open, stat, seek)
    than actually transferring data. Requires a minimum time threshold to ignore
    tiny, fast jobs where metadata inherently dominates.
    """
    print("\n--- 2. Metadata Hammers (Meta Time > Transfer Time) ---")
    
    df['total_transfer_time'] = df['POSIX_F_READ_TIME'] + df['POSIX_F_WRITE_TIME']
    
    offenders = df[
        (df['POSIX_F_META_TIME'] > min_meta_time_sec) & 
        (df['POSIX_F_META_TIME'] > df['total_transfer_time'])
    ]
    
    if offenders.empty:
        print("[CLEAR] No metadata-bound jobs found.")
    else:
        print(f"[WARNING] Found {len(offenders)} jobs suffocating on metadata.")
        display_cols = ['job_id', 'uid', 'exe', 'POSIX_OPENS', 'POSIX_STATS', 'POSIX_F_META_TIME', 'total_transfer_time']
        
        offenders = offenders.sort_values(by='POSIX_F_META_TIME', ascending=False)
        
        console_df = offenders[display_cols].copy()
        console_df['POSIX_F_META_TIME'] = console_df['POSIX_F_META_TIME'].round(1).astype(str) + 's'
        console_df['total_transfer_time'] = console_df['total_transfer_time'].round(1).astype(str) + 's'
        
        print(console_df.head(10).to_string(index=False))
        offenders[display_cols].to_csv('report_metadata.csv', index=False)
        print("-> Full list saved to report_metadata.csv")

def analyze_mpi_io(df: pd.DataFrame, volume_threshold_gb: float = 1.0):
    """
    Finds jobs using MPI-IO to move significant data, but failing to use
    Collective I/O operations, meaning they are likely bottlenecking.
    """
    print("\n--- 3. Un-optimized MPI-IO (Independent I/O Only) ---")
    
    volume_threshold_bytes = volume_threshold_gb * (1024**3)
    
    # Jobs that wrote significant data using MPI-IO
    mpi_writers = df[df['MPIIO_BYTES_WRITTEN'] > volume_threshold_bytes]
    
    # Of those, which ones did exactly zero collective writes?
    offenders = mpi_writers[mpi_writers['MPIIO_COLL_WRITES'] == 0]
    
    if offenders.empty:
        print("[CLEAR] All heavy MPI-IO users are utilizing Collective operations.")
    else:
        print(f"[WARNING] Found {len(offenders)} MPI jobs ignoring Collective I/O.")
        display_cols = ['job_id', 'uid', 'exe', 'nprocs', 'MPIIO_BYTES_WRITTEN', 'MPIIO_INDEP_WRITES']
        
        offenders = offenders.sort_values(by='MPIIO_BYTES_WRITTEN', ascending=False)
        
        console_df = offenders[display_cols].copy()
        console_df['MPIIO_BYTES_WRITTEN'] = (console_df['MPIIO_BYTES_WRITTEN'] / (1024**3)).round(2).astype(str) + ' GB'
        
        print(console_df.head(10).to_string(index=False))
        offenders[display_cols].to_csv('report_mpiio.csv', index=False)
        print("-> Full list saved to report_mpiio.csv")


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
        
        analyze_tiny_io(master_df, volume_threshold_gb=args.tiny_io_gb, size_threshold_kb=args.tiny_io_kb)
        analyze_metadata_hammers(master_df, min_meta_time_sec=args.meta_sec)
        analyze_mpi_io(master_df, volume_threshold_gb=args.tiny_io_gb)
        
        print("\n[SUCCESS] Analysis complete.")
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
