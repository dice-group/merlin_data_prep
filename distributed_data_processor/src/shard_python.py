import gzip, os, time, argparse, shutil, subprocess
from multiprocessing import Pool, cpu_count

DEFAULT_CHUNK_MB = 512

def process_file(args):
    path, src, dst, chunk_bytes, pigz_path, pigz_threads, compresslevel = args
    t0 = time.time()
    rel = os.path.relpath(path, src)
    out_dir = os.path.join(dst, os.path.dirname(rel))
    os.makedirs(out_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(path))[0]
    total = 0
    cur = 0
    parts = 0
    tmp_path = os.path.join(out_dir, f".{name}.tmp.{os.getpid()}.jsonl.gz")
    pigz = pigz_path

    def open_reader():
        if pigz:
            proc = subprocess.Popen([pigz, "-dc", path], stdout=subprocess.PIPE)
            return proc, proc.stdout
        return None, gzip.open(path, "rb")

    def open_writer(out_path):
        if pigz:
            out_fh = open(out_path, "wb")
            cmd = [pigz, f"-{compresslevel}", "-c"]
            if pigz_threads:
                cmd += ["-p", str(pigz_threads)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=out_fh)
            return ("pigz", proc, out_fh)
        return ("gzip", gzip.open(out_path, "wb", compresslevel=compresslevel), None)

    def close_writer(kind, handle, out_fh):
        if kind == "pigz":
            handle.stdin.close()
            rc = handle.wait()
            out_fh.close()
            if rc != 0:
                raise RuntimeError(f"pigz compression failed with code {rc}")
        else:
            handle.close()

    def part_path(idx):
        return os.path.join(out_dir, f"{name}_part_{idx:05d}.jsonl.gz")

    proc = None
    fin = None
    try:
        proc, fin = open_reader()
        kind, out, out_fh = open_writer(tmp_path)
        for line in fin:
            line_len = len(line)
            if cur + line_len > chunk_bytes and cur > 0:
                close_writer(kind, out, out_fh)
                if parts == 0:
                    os.replace(tmp_path, part_path(0))
                parts += 1
                kind, out, out_fh = open_writer(part_path(parts))
                cur = 0
            out.write(line) if kind == "gzip" else out.stdin.write(line)
            cur += line_len
            total += line_len
        close_writer(kind, out, out_fh)
    finally:
        if fin:
            fin.close()
        if proc:
            if proc.stdout:
                proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"pigz decompression failed with code {rc}")

    if parts == 0:
        os.replace(tmp_path, os.path.join(out_dir, os.path.basename(path)))
        print(f"[COPY] {rel} ({total/1e6:.1f} MB uncompressed) in {time.time()-t0:.1f}s")
        return

    print(f"[SHARD] {rel} -> {parts+1} parts in {time.time()-t0:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Shard or copy .gz files by size.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Input directory")
    group.add_argument("--file", "-f", help="Single .gz file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--chunk-mb", type=float, default=DEFAULT_CHUNK_MB, help="Uncompressed chunk size in MB")
    parser.add_argument("--compresslevel", type=int, default=1, help="Gzip compression level (1 fastest, 9 smallest)")
    parser.add_argument("--pigz", action="store_true", help="Use pigz for faster (de)compression if available")
    parser.add_argument("--pigz-threads", type=int, default=0, help="pigz thread count (0 = pigz default)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    chunk_bytes = max(1, int(args.chunk_mb * 1024 * 1024))
    pigz_path = shutil.which("pigz") if args.pigz else None
    if args.pigz and not pigz_path:
        print("[WARN] --pigz requested but pigz not found; falling back to gzip.")

    # single-file mode (Slurm array)
    if args.file:
        src = os.path.dirname(args.file)
        process_file((args.file, src, args.output, chunk_bytes, pigz_path, args.pigz_threads, args.compresslevel))
        return

    # directory mode
    src = args.input
    files = [
        os.path.join(r, f)
        for r, _, fs in os.walk(src)
        for f in fs if f.endswith(".gz")
    ]

    print(f"[INFO] Found {len(files)} files")
    start = time.time()

    with Pool(min(cpu_count(), 16)) as p:
        for _ in p.imap_unordered(
            process_file, [(f, src, args.output, chunk_bytes, pigz_path, args.pigz_threads, args.compresslevel) for f in files]
        ):
            pass

    print(f"[TOTAL] Done all in {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
