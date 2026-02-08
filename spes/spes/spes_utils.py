from spes.spes import federated_pb2
from spes.spes import federated_pb2_grpc
import time

def upload_weights_in_chunks(stub, peer_id, step, weights_bytes, chunk_size=1936*1024*1024):
    total_chunks = (len(weights_bytes) + chunk_size - 1) // chunk_size
    def req_iter():
        for i in range(total_chunks):
            chunk = weights_bytes[i*chunk_size:(i+1)*chunk_size]
            yield federated_pb2.UploadChunkRequest(
                peer_id=peer_id,
                step=step,
                chunk_id=i,
                total_chunks=total_chunks,
                chunk_data=chunk
            )
    resp = stub.UploadChunk(req_iter())
    assert resp.success

def download_weights_in_chunks(stub, peer_id, step, chunk_size=1936*1024*1024):
    chunks = []
    chunk_id = 0
    while True:
        resp = stub.DownloadChunk(
            federated_pb2.DownloadChunkRequest(
                peer_id=peer_id,
                step=step,
                chunk_id=chunk_id
            )
        )
        if not resp.ready:
            time.sleep(20)
            continue
        chunks.append(resp.chunk_data)
        if resp.last_chunk:
            break
        chunk_id += 1
    return b''.join(chunks)