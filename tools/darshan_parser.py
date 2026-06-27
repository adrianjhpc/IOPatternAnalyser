import os
import argparse
from pathlib import Path
import pandas as pd
import darshan
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_single_log(log_path, posix_metrics, mpiio_metrics, stdio_metrics, h5_metrics, nc_metrics):
    """Isolated function to parse a single log. Runs in a separate process."""
    try:
        report = darshan.DarshanReport(str(log_path))
        
        start_time = int(report.metadata.get('start_time', 0))
        end_time = int(report.metadata.get('end_time', 0))
        runtime = end_time - start_time if end_time > start_time else 0

        record_summary = {
            'source_file': log_path.name,
            'job_id': report.metadata.get('jobid', 'unknown'),
            'uid': report.metadata.get('uid', 'unknown'),
            'exe': report.metadata.get('exe', 'unknown'),
            'nprocs': int(report.metadata.get('nprocs', 1)),
            'runtime_sec': runtime
        }

        record_summary.update(extract_module_metrics(report, 'POSIX', posix_metrics))
        record_summary.update(extract_module_metrics(report, 'MPI-IO', mpiio_metrics))
        record_summary.update(extract_module_metrics(report, 'STDIO', stdio_metrics))
        record_summary.update(extract_module_metrics(report, 'H5', h5_metrics))
        record_summary.update(extract_module_metrics(report, 'NC', nc_metrics))

        return record_summary
    except Exception as e:
        return {"error": f"Failed to parse {log_path.name}: {e}"}

def extract_module_metrics(report: darshan.DarshanReport, module_name: str, metrics: list) -> dict:
    """
    Safely extracts requested metrics from both 'counters' (integers) 
    and 'fcounters' (floats/timers) DataFrames for a given module.
    """
    results = {metric: 0.0 for metric in metrics}
    
    if module_name not in report.modules:
        return results

    try:
        report.mod_read_all_records(module_name)
        if module_name not in report.records:
            return results
        
        mod_record = report.records[module_name]
        if not hasattr(mod_record, 'to_df'):
            return results
            
        data = mod_record.to_df()
        
        # PyDarshan structure: dict containing 'counters' and 'fcounters'
        if isinstance(data, dict):
            counters_df = data.get('counters', pd.DataFrame())
            fcounters_df = data.get('fcounters', pd.DataFrame())
        elif isinstance(data, pd.DataFrame):
            # Fallback for older versions/weird structures
            counters_df = data
            fcounters_df = pd.DataFrame()
        else:
            return results

        # Sum the metrics if they exist in either DataFrame
        for metric in metrics:
            if not counters_df.empty and metric in counters_df.columns:
                results[metric] = counters_df[metric].sum()
            elif not fcounters_df.empty and metric in fcounters_df.columns:
                results[metric] = fcounters_df[metric].sum()
                
    except Exception:
        # Silently catch Pandas indexing/type errors to prevent crashing the batch
        pass

    return results

def aggregate_darshan_logs(base_dir: str) -> pd.DataFrame:
    """
    Recursively finds and parses .darshan logs, extracting extensive
    metrics for system-wide I/O pattern analysis.
    """
    base_path = Path(base_dir)
    log_files = list(base_path.rglob("*.darshan"))

    if not log_files:
        print(f"No .darshan files found in {base_dir}")
        return pd.DataFrame()

    print(f"Found {len(log_files)} Darshan logs. Beginning comprehensive parsing...")
    summary_records = []

    # Define the exact metrics we want to pull per module
    posix_metrics = [
        'POSIX_BYTES_READ', 'POSIX_BYTES_WRITTEN', 
        'POSIX_OPENS', 'POSIX_STATS', 'POSIX_SEEKS',
        'POSIX_READS', 'POSIX_WRITES',
        'POSIX_CONSEC_READS', 'POSIX_CONSEC_WRITES', 
        'POSIX_SEQ_READS', 'POSIX_SEQ_WRITES',
        'POSIX_F_READ_TIME', 'POSIX_F_WRITE_TIME', 'POSIX_F_META_TIME'
    ]
    
    mpiio_metrics = [
        'MPIIO_BYTES_READ', 'MPIIO_BYTES_WRITTEN',
        'MPIIO_INDEP_READS', 'MPIIO_INDEP_WRITES',
        'MPIIO_COLL_READS', 'MPIIO_COLL_WRITES',
        'MPIIO_F_READ_TIME', 'MPIIO_F_WRITE_TIME', 'MPIIO_F_META_TIME'
    ]
    
    stdio_metrics = [
        'STDIO_BYTES_READ', 'STDIO_BYTES_WRITTEN',
        'STDIO_OPENS', 'STDIO_READS', 'STDIO_WRITES',
        'STDIO_F_META_TIME', 'STDIO_F_READ_TIME', 'STDIO_F_WRITE_TIME'
    ]

    h5_metrics = ['H5_BYTES_READ', 'H5_BYTES_WRITTEN', 'H5_F_META_TIME']
    nc_metrics = ['NC_BYTES_READ', 'NC_BYTES_WRITTEN']

# Use a ProcessPoolExecutor to isolate C-level crashes
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all jobs to the process pool
        futures = {
            executor.submit(
                parse_single_log, log_path, posix_metrics, mpiio_metrics, stdio_metrics, h5_metrics, nc_metrics
            ): log_path for log_path in log_files if log_path.stat().st_size >= 1024
        }

        for future in as_completed(futures):
            log_path = futures[future]
            try:
                result = future.result()
                if "error" in result:
                    print(result["error"])
                else:
                    summary_records.append(result)
            except Exception as e: # This catches standard Python errors in the worker
                print(f"Worker raised error for {log_path.name}: {e}")
            except (ProcessLookupError, BrokenPipeError, EOFError) as e: 
                # This catches the worker dying due to a C-level abort
                print(f"CRITICAL: Worker process violently crashed on {log_path.name}. File is severely corrupted.")

    if summary_records:
        master_df = pd.DataFrame(summary_records)
        print("Aggregated parsing complete!")
        return master_df
    else:
        print("Parsing complete, but no data was extracted.")
        return pd.DataFrame()

# ==========================================
# Command Line Execution & Analysis
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Darshan logs for whole-system I/O analysis.")
    parser.add_argument("log_dir", type=str, help="Path to the directory containing .darshan logs")
    parser.add_argument("-o", "--output", type=str, default="darshan_comprehensive.csv",
                        help="Output file path. Use .csv or .parquet extension.")

    args = parser.parse_args()

    df = aggregate_darshan_logs(args.log_dir)

    if not df.empty:
        # Save dataset
        if args.output.endswith('.parquet'):
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Aggregated dataset saved to: {args.output}")

        # --- Quick System-Wide Sanity Check ---
        print("\n--- System-Wide I/O Health Snapshot ---")
        
        # Calculate cluster-level aggregates
        total_runtime_hrs = df['runtime_sec'].sum() / 3600
        gb_divisor = 1024**3
        total_posix_read = df['POSIX_BYTES_READ'].sum() / gb_divisor
        total_posix_write = df['POSIX_BYTES_WRITTEN'].sum() / gb_divisor
        
        # Avoid division by zero
        total_posix_reads = df['POSIX_READS'].sum()
        total_posix_writes = df['POSIX_WRITES'].sum()
        avg_read_size_kb = (df['POSIX_BYTES_READ'].sum() / total_posix_reads / 1024) if total_posix_reads > 0 else 0
        avg_write_size_kb = (df['POSIX_BYTES_WRITTEN'].sum() / total_posix_writes / 1024) if total_posix_writes > 0 else 0

        total_io_time = df['POSIX_F_READ_TIME'].sum() + df['POSIX_F_WRITE_TIME'].sum()
        total_meta_time = df['POSIX_F_META_TIME'].sum()

        print(f"Total Jobs Parsed:      {len(df)}")
        print(f"Total Compute Hours:    {total_runtime_hrs:,.2f} hrs")
        print(f"Total POSIX Read:       {total_posix_read:,.2f} GB")
        print(f"Total POSIX Write:      {total_posix_write:,.2f} GB")
        print(f"Avg POSIX Read Size:    {avg_read_size_kb:,.2f} KB")
        print(f"Avg POSIX Write Size:   {avg_write_size_kb:,.2f} KB")
        
        print(f"\nTime Spent in I/O (Cumulative App-level Time):")
        print(f"Data Transfer Time:     {total_io_time:,.2f} sec")
        print(f"Metadata Ops Time:      {total_meta_time:,.2f} sec")
        
        if total_meta_time > total_io_time:
            print("\n[WARNING] System-wide metadata time exceeds data transfer time. Expect filesystem latency issues.")
