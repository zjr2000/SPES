import os
import glob
import numpy as np
from transformers import AutoTokenizer
import gzip
import json
import pyarrow.parquet as pq
from multiprocessing import Pool
import random
import zstandard as zstd
import io

def read_lines(path: str, text_field: str = "text"):
    """
    Generator yielding `text_field` from various file types.

    Parameters
    ----------
    path : str
        File path.
    text_field : str
        Key whose value will be yielded.

    Yields
    ------
    str
        The stripped text for each record.
    """
    # 1) Parquet -------------------------------------------------------------
    if path.endswith(".parquet"):
        print(f"Reading parquet: {path}")
        table = pq.read_table(path, columns=[text_field])
        for val in table[text_field].to_pylist():
            if isinstance(val, str):
                text = val.strip()
                if text:
                    yield text

    # 2) jsonl.zst -----------------------------------------------------------
    elif path.endswith(".jsonl.zst"):
        print(f"Reading zstd-compressed jsonl: {path}")
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader, \
                 io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for i, line in enumerate(text_stream):
                    try:
                        obj = json.loads(line)
                        text = (obj.get(text_field) or "").strip()
                        if text:
                            yield text
                    except Exception as e:
                        print(f"Error parsing line {i} in {path}: {e}")
                        continue

    # 3) jsonl.gz ------------------------------------------------------------
    elif path.endswith(".jsonl.gz") or path.endswith(".json.gz"):
        print(f"Reading gzip-compressed jsonl: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    text = (obj.get(text_field) or "").strip()
                    if text:
                        yield text
                except Exception as e:
                    print(f"Error parsing line {i} in {path}: {e}")
                    continue

    # 5) 其他：假定为 jsonl 或 json（未压缩） -------------------------------
    else:
        print(f"Reading (plain?) json/jsonl: {path}")
        # 判断是否是逐行 JSONL：如果文件很大或首行以 `{` 结尾概率大
        with open(path, "rt", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            is_jsonl = first_char not in ["[", "{"]  # 简单启发
            if is_jsonl:
                for i, line in enumerate(f):
                    try:
                        obj = json.loads(line)
                        text = (obj.get(text_field) or "").strip()
                        if text:
                            yield text
                    except Exception as e:
                        print(f"Error parsing line {i} in {path}: {e}")
                        continue
            else:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        iterable = data
                    else:
                        iterable = [data]
                    for obj in iterable:
                        if isinstance(obj, dict):
                            text = (obj.get(text_field) or "").strip()
                            if text:
                                yield text
                except Exception as e:
                    print(f"Error loading {path}: {e}")

def process_files(
    file_list,
    tokenizer_name_or_path,
    output_prefix,
    text_field="text",
    batch_size=10000,
    max_shard_bytes=2 * 1024 * 1024 * 1024,  # 2GB
    add_special_tokens=False,
    add_eos_token=True,
    dtype="uint32",
    process_idx=0
):
    print(f"Process {process_idx} starting. Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    eos_token_id = tokenizer.eos_token_id

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    token_batches = []
    current_tokens = 0
    shard_idx = 1
    TOKEN_BYTES = np.dtype(dtype).itemsize

    def save_shard(token_batches, shard_idx):
        all_tokens = np.concatenate(token_batches)
        num_tokens = len(all_tokens)
        out_path = f"{output_prefix}{process_idx:03d}_{shard_idx:06d}.npy"
        mmap_arr = np.memmap(out_path, mode="w+", dtype=dtype, shape=(num_tokens,))
        mmap_arr[:] = all_tokens
        mmap_arr.flush()
        del mmap_arr
        print(f"Process {process_idx}: Saved shard {shard_idx} ({num_tokens} tokens, {num_tokens*TOKEN_BYTES/1024/1024/1024:.2f} GB) to {out_path}")

    total_files = len(file_list)
    total_batches = 0
    total_texts = 0
    total_tokenized = 0

    for file_idx, path in enumerate(file_list):
        print(f"Process {process_idx}: Start processing file {file_idx+1}/{total_files}: {path}")
        batch = []
        file_text_count = 0
        file_token_count = 0
        for text in read_lines(path, text_field=text_field):
            text = text.replace("<extra_id_1>", "")
            batch.append(text)
            file_text_count += 1
            if len(batch) >= batch_size:
                random.shuffle(batch)
                batch_token_ids = []
                for text_item in batch:
                    try:
                        tokens = tokenizer.encode(text_item, add_special_tokens=add_special_tokens)
                        if add_eos_token and eos_token_id is not None:
                            tokens.append(eos_token_id)
                        batch_token_ids.append(np.array(tokens, dtype=dtype))
                        file_token_count += len(tokens)
                        total_tokenized += len(tokens)
                    except Exception as e:
                        print(f"Process {process_idx}: Tokenize error: {e} (text={text_item[:50]})")
                        continue
                token_batches.extend(batch_token_ids)
                current_tokens += sum(len(arr) for arr in batch_token_ids)
                total_batches += 1
                total_texts += len(batch)
                # print(f"Process {process_idx}: Finished batch {total_batches} ({len(batch)} texts, {sum(len(arr) for arr in batch_token_ids)} tokens)")
                batch.clear()
                current_shard_bytes = current_tokens * TOKEN_BYTES
                if current_shard_bytes >= max_shard_bytes:
                    print(f"Process {process_idx}: Shard size {current_shard_bytes/1024/1024/1024:.2f} GB reached, saving...")
                    save_shard(token_batches, shard_idx)
                    shard_idx += 1
                    token_batches.clear()
                    current_tokens = 0
        # 处理最后不足batch_size的部分
        if batch:
            random.shuffle(batch)
            batch_token_ids = []
            for text_item in batch:
                try:
                    tokens = tokenizer.encode(text_item, add_special_tokens=add_special_tokens)
                    if add_eos_token and eos_token_id is not None:
                        tokens.append(eos_token_id)
                    batch_token_ids.append(np.array(tokens, dtype=dtype))
                    file_token_count += len(tokens)
                    total_tokenized += len(tokens)
                except Exception as e:
                    print(f"Process {process_idx}: Tokenize error: {e} (text={text_item[:50]})")
                    continue
            token_batches.extend(batch_token_ids)
            current_tokens += sum(len(arr) for arr in batch_token_ids)
            total_batches += 1
            total_texts += len(batch)
            batch.clear()
            current_shard_bytes = current_tokens * TOKEN_BYTES
            if current_shard_bytes >= max_shard_bytes:
                print(f"Process {process_idx}: Shard size {current_shard_bytes/1024/1024/1024:.2f} GB reached, saving...")
                save_shard(token_batches, shard_idx)
                shard_idx += 1
                token_batches.clear()
                current_tokens = 0

        print(f"Process {process_idx}: Finished file {file_idx+1}/{total_files}: {path} ({file_text_count} texts, {file_token_count} tokens)")

        # try:
        #     os.remove(path)
        #     print(f"Process {process_idx}: Deleted source file: {path}")
        # except Exception as e:
        #     print(f"Process {process_idx}: Failed to delete source file {path}: {e}")


    if token_batches:
        print(f"Process {process_idx}: Saving final shard {shard_idx} ({current_tokens} tokens)...")
        save_shard(token_batches, shard_idx)

    print(f"Process {process_idx}: All done. Total files: {total_files}, total batches: {total_batches}, total texts: {total_texts}, total tokens: {total_tokenized}")

def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main(
    file_glob,
    tokenizer_name_or_path,
    output_prefix,
    text_field="text",
    processes=1,
    files_per_process=None,
    batch_size=10000,
    max_shard_bytes=2 * 1024 * 1024 * 1024,
    dtype="uint32"
):
    files = sorted(glob.glob(file_glob))
    if files_per_process:
        file_chunks = [files[i:i+files_per_process] for i in range(0, len(files), files_per_process)]
    else:
        file_chunks = chunk_list(files, processes)
    print(f"Total files: {len(files)}, processes: {processes}, file_chunks: {len(file_chunks)}")

    args_list = []
    for idx, file_list in enumerate(file_chunks):
        print(f"Assigning {len(file_list)} files to process {idx}")
        args_list.append((
            file_list,
            tokenizer_name_or_path,
            output_prefix,
            text_field,
            batch_size,
            max_shard_bytes,
            False,   # add_special_tokens
            True,    # add_eos_token
            dtype,
            idx
        ))

    if processes > 1:
        print(f"Spawning {processes} processes...")
        with Pool(processes) as pool:
            pool.starmap(process_files, args_list)
    else:
        for args in args_list:
            process_files(*args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_glob", type=str, default="./data/*")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument("--output_prefix", type=str, default="./tokenized/qwen_tokens_")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--files_per_process", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--max_shard_bytes", type=int, default=4 * 1024 * 1024 * 1024, help="每个shard最大字节数，默认2GB")
    parser.add_argument("--dtype", type=str, default="uint32")
    args = parser.parse_args()

    main(
        args.file_glob,
        args.tokenizer_name_or_path,
        args.output_prefix,
        args.text_field,
        args.processes,
        args.files_per_process,
        args.batch_size,
        args.max_shard_bytes,
        args.dtype
    )
