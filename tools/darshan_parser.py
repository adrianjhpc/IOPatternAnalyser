import os
from pathlib import Path
import pandas as pd
import darshan

def aggregate_darshan_logs(base_dir: str) -> pd.DataFrame:
    """
    Recursively finds and parses .darshan logs. 
    Aggregates POSIX, MPI-IO, HDF5, NetCDF, and STDIO metrics inside the loop to save memory,
    returning a single summary DataFrame where each row represents one job.
    """
    base_path = Path(base_dir)
    log_files = list(base_path.rglob("*.darshan"))
    
    if not log_files:
        print(f"No .darshan files found in {base_dir}")
        return pd.DataFrame()

    print(f"Found {len(log_files)} Darshan logs. Beginning aggregated parsing...")
    
    # This will hold one dictionary (row) per log file
    summary_records = []

    for log_path in log_files:
        try:
            # Initialize the report
            report = darshan.DarshanReport(str(log_path), read_all=True)
            
            # 1. Setup a base dictionary for this specific job's summary
            record_summary = {
                'source_file': log_path.name,
                'job_id': report.metadata.get('jobid', 'unknown'),
                'exe': report.metadata.get('exe', 'unknown'),
                'nprocs': report.metadata.get('nprocs', 1),
                'posix_read_bytes': 0, 'posix_write_bytes': 0,
                'mpiio_read_bytes': 0, 'mpiio_write_bytes': 0,
                'stdio_read_bytes': 0, 'stdio_write_bytes': 0,
                'hdf5_read_bytes': 0, 'hdf5_write_bytes': 0,
                'netcdf_read_bytes': 0, 'netcdf_write_bytes': 0,
            }
            
            # 2. Extract POSIX Data
            if 'POSIX' in report.records:
                posix_df = report.records['POSIX'].to_df()
                if 'POSIX_BYTES_READ' in posix_df.columns:
                    record_summary['posix_read_bytes'] = posix_df['POSIX_BYTES_READ'].sum()
                if 'POSIX_BYTES_WRITTEN' in posix_df.columns:
                    record_summary['posix_write_bytes'] = posix_df['POSIX_BYTES_WRITTEN'].sum()

            # 3. Extract MPI-IO Data
            if 'MPI-IO' in report.records:
                mpi_df = report.records['MPI-IO'].to_df()
                if 'MPIIO_BYTES_READ' in mpi_df.columns:
                    record_summary['mpiio_read_bytes'] = mpi_df['MPIIO_BYTES_READ'].sum()
                if 'MPIIO_BYTES_WRITTEN' in mpi_df.columns:
                    record_summary['mpiio_write_bytes'] = mpi_df['MPIIO_BYTES_WRITTEN'].sum()

            # 4. Extract STDIO Data
            if 'STDIO' in report.records:
                stdio_df = report.records['STDIO'].to_df()
                if 'STDIO_BYTES_READ' in stdio_df.columns:
                    record_summary['stdio_read_bytes'] = stdio_df['STDIO_BYTES_READ'].sum()
                if 'STDIO_BYTES_WRITTEN' in stdio_df.columns:
                    record_summary['stdio_write_bytes'] = stdio_df['STDIO_BYTES_WRITTEN'].sum()
            
            # 5. Extract HDF5 Data
            if 'H5' in report.records:
                hdf5_df = report.records['H5'].to_df()
                if 'H5_BYTES_READ' in hdf5_df.columns:
                    record_summary['hdf5_read_bytes'] = hdf5_df['H5_BYTES_READ'].sum()
                if 'H5_BYTES_WRITTEN' in hdf5_df.columns:
                    record_summary['hdf5_write_bytes'] = hdf5_df['H5_BYTES_WRITTEN'].sum()

            # 6. Extract NetCDF Data
            if 'NC' in report.records:
                netcdf_df = report.records['NC'].to_df()
                if 'NC_BYTES_READ' in netcdf_df.columns:
                    record_summary['netcdf_read_bytes'] = netcdf_df['NC_BYTES_READ'].sum()
                if 'NC_BYTES_WRITTEN' in netcdf_df.columns:
                    record_summary['netcdf_write_bytes'] = netcdf_df['NC_BYTES_WRITTEN'].sum()

            # Append the lightweight summary to our master list
            summary_records.append(record_summary)
            
        except Exception as e:
            print(f"Error parsing {log_path.name}: {e}")

    # Convert the list of summaries directly into a DataFrame
    if summary_records:
        master_df = pd.DataFrame(summary_records)
        print("Aggregated parsing complete!")
        return master_df
    else:
        print("Parsing complete, but no data was extracted.")
        return pd.DataFrame()

# ==========================================
# Usage & Analysis
# ==========================================
if __name__ == "__main__":
    log_directory = "./darshan_logs"
    
    # Process the logs
    df = aggregate_darshan_logs(log_directory)
    
    if not df.empty:
        print("\n--- Aggregated I/O Analysis ---")
        
        # Calculate totals across the entire dataset in Gigabytes
        gb_divisor = 1024**3
        
        print("\nTotal I/O by Interface (GB):")
        print(f"POSIX Read:   {df['posix_read_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"POSIX Write:  {df['posix_write_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"MPI-IO Read:  {df['mpiio_read_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"MPI-IO Write: {df['mpiio_write_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"STDIO Read:   {df['stdio_read_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"STDIO Write:  {df['stdio_write_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"HDF5 Read:   {df['hdf5_read_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"HDF5 Write:  {df['hdf5_write_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"NetCDF Read:   {df['netcdf_read_bytes'].sum() / gb_divisor:,.2f} GB")
        print(f"NetCDF Write:  {df['netcdf_write_bytes'].sum() / gb_divisor:,.2f} GB")        

        # Determine which application pushed the most MPI-IO traffic
        print("\nTop 3 Executables by MPI-IO Write Volume:")
        top_mpi = df.groupby('exe')['mpiio_write_bytes'].sum().sort_values(ascending=False).head(3)
        # Format the output cleanly
        for exe, bytes_written in top_mpi.items():
            print(f"{exe}: {bytes_written / gb_divisor:,.2f} GB")
