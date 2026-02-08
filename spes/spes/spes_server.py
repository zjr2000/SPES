import grpc
from concurrent import futures
import federated_pb2
import federated_pb2_grpc
import torch
import io
import argparse
import torch.nn as nn
from tqdm import tqdm
import re
import copy
import gc


CHUNK_SIZE = 1936 * 1024 * 1024

class FederatedServer(federated_pb2_grpc.FederatedServerServicer):
    def __init__(self, total_peers, keep_steps=1, num_train_experts_per_node=1):
        self.total_peers = total_peers
        self.chunk_uploads = {}
        self.uploads = {}      # {step: {peer_id: state_dict_bytes}}
        self.aggregated = {}   # {step: state_dict_bytes}
        self.keep_steps = keep_steps
        self.num_train_experts_per_node = num_train_experts_per_node

        # === Nesterov Momentum Buffers ===
        self.prev_global_params = {}   # 上一轮的全局参数
        self.momentum_buffers = {}     # 动量缓存

    def UploadChunk(self, request_iterator, context):
        for req in request_iterator:
            step = int(req.step)
            peer_id = int(req.peer_id)
            chunk_id = int(req.chunk_id)
            total_chunks = int(req.total_chunks)

            if step not in self.chunk_uploads:
                self.chunk_uploads[step] = {}
            if peer_id not in self.chunk_uploads[step]:
                self.chunk_uploads[step][peer_id] = {}
            self.chunk_uploads[step][peer_id][chunk_id] = req.chunk_data

            print(f"Received chunk {chunk_id+1}/{total_chunks} from peer {peer_id} step {step}")

            # 检查是否所有分片都收齐
            if len(self.chunk_uploads[step][peer_id]) == total_chunks:
                # 拼接
                chunks = [self.chunk_uploads[step][peer_id][i] for i in range(total_chunks)]
                full_bytes = b''.join(chunks)
                if step not in self.uploads:
                    self.uploads[step] = {}
                self.uploads[step][peer_id] = full_bytes
                print(f"[UploadChunk] Collected {len(self.uploads[step])}/{self.total_peers} uploads for step {step}")
                del self.chunk_uploads[step][peer_id]
                # 如果该step所有peer都上传完毕，聚合
                if len(self.uploads[step]) == self.total_peers:
                    self.aggregate_step(step)

        return federated_pb2.UploadChunkResponse(success=True)

    def DownloadChunk(self, request, context):
        step = request.step
        peer_id = int(request.peer_id)
        if step in self.aggregated:
            full_bytes = self.aggregated[step]
            chunk_id = request.chunk_id
            start = chunk_id * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk_data = full_bytes[start:end]
            last_chunk = end >= len(full_bytes)
            return federated_pb2.DownloadChunkResponse(
                chunk_data=chunk_data,
                last_chunk=last_chunk,
                ready=True
            )
        else:
            return federated_pb2.DownloadChunkResponse(
                chunk_data=b'',
                last_chunk=True,
                ready=False
            )

    def aggregate_step(self, step):
        print(f"[Aggregate] All peers uploaded for step {step}, starting aggregation.")
        # 1. Load all state_dicts
        print(f"[Aggregate] Loading state_dicts for peers: {list(self.uploads[step].keys())}")
        state_dicts = {}
        for pid in self.uploads[step]:
            try:
                state_dicts[pid] = torch.load(io.BytesIO(self.uploads[step][pid]), map_location='cpu')
                print(f"[Aggregate] Loaded state_dict for peer {pid}")
            except Exception as e:
                print(f"[Error] Failed to load state_dict for peer {pid}: {e}")

        print(f"[Aggregate] Number of state_dicts loaded: {len(state_dicts)}")

        all_module_keys = set()
        for sd in state_dicts.values():
            all_module_keys.update(sd.keys())
        all_module_keys = list(all_module_keys)
        print(all_module_keys)

        resulted_state_dicts = {}
        with torch.no_grad():
            for key in all_module_keys:
                if "ffn.experts.mlp" in key:
                    expert_idx = int(key.split('.')[-1])
                    corresponding_peer_id = expert_idx // self.num_train_experts_per_node
                    resulted_state_dicts[key] = state_dicts[corresponding_peer_id][key]
                    print(f"[Aggregate] {key}: using weight from peer {corresponding_peer_id}")
                else:
                    tensors = [sd[key].float() for sd in state_dicts.values()]
                    avg_param = torch.stack(tensors, dim=0).mean(dim=0)
                    resulted_state_dicts[key] = avg_param

            print(f"[Aggregate] Aggregated state_dict ready.")

            buf = io.BytesIO()
            torch.save(resulted_state_dicts, buf)
            buf.seek(0)
            self.aggregated[step] = buf.read()
            del resulted_state_dicts  # 释放聚合结果
            buf.close()               # 关闭buffer

            print(f"[Store] Aggregated weights for at step {step} stored.")

        # 释放state_dicts
        state_dicts.clear()
        del state_dicts
        gc.collect()  # 强制回收内存

        print(f"[Aggregate] Aggregation finished for step {step}. Aggregated weights ready for download.")
        del self.uploads[step]

        # Only keep the most recent `keep_steps` aggregated weights
        for old_step in list(self.aggregated.keys()):
            if old_step < step - self.keep_steps:
                print(f"[Cleanup] Removing old aggregated weights for step {old_step}")
                del self.aggregated[old_step]
                gc.collect()  # 清理后再收集


def serve(total_peers, port=50051, num_train_experts_per_node=1):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), 
        options=[
            ('grpc.max_send_message_length', 1937*1024*1024),   # 1GB
            ('grpc.max_receive_message_length', 1937*1024*1024), # 1GB
            ('grpc.max_metadata_size', 120 * 1024)
        ]
    )
    federated_pb2_grpc.add_FederatedServerServicer_to_server(FederatedServer(total_peers, num_train_experts_per_node=num_train_experts_per_node), server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"[Server] Server started on port {port} for {total_peers} peers.")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_peers', type=int, default=1, help="总的client数")
    parser.add_argument('--num_train_experts_per_node', type=int, default=1)
    parser.add_argument('--port', type=int, default=50051)
    args = parser.parse_args()
    serve(args.total_peers, args.port, args.num_train_experts_per_node)
