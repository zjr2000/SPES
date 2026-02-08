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
    def __init__(self, total_peers, keep_steps=1, num_train_experts_per_node=1,
                 merge_interval=200, merge_alpha_start=0.05, merge_decay_steps=2000):
        self.total_peers = total_peers
        self.chunk_uploads = {}
        self.uploads = {}      # {step: {peer_id: state_dict_bytes}}
        self.aggregated = {}   # {step: state_dict_bytes}
        self.keep_steps = keep_steps
        self.num_train_experts_per_node = num_train_experts_per_node

        # merging 控制参数
        self.merge_interval = merge_interval
        self.merge_alpha_start = merge_alpha_start
        self.merge_decay_steps = merge_decay_steps

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

        resulted_state_dicts = {}
        with torch.no_grad():
            for key in all_module_keys:
                if "ffn.experts.mlp" in key:
                    expert_idx = int(key.split('.')[-1])
                    corresponding_peer_id = expert_idx // self.num_train_experts_per_node
                    resulted_state_dicts[key] = state_dicts[corresponding_peer_id][key]
                    # print(f"[Aggregate] {key}: using weight from peer {corresponding_peer_id}")
                else:
                    tensors = [sd[key].float() for sd in state_dicts.values()]
                    avg_param = torch.stack(tensors, dim=0).mean(dim=0)
                    resulted_state_dicts[key] = avg_param

            # merge 控制逻辑
            if step % self.merge_interval == 0:
                progress = min(step / self.merge_decay_steps, 1.0)
                alpha_now = self.merge_alpha_start * (1.0 - progress)

                if alpha_now > 0:
                    print(f"[Merge] Step {step}: alpha={alpha_now:.6f}")
                    resulted_state_dicts = merge_experts_task_vector_topk_cosine_w1_per_layer(
                        resulted_state_dicts, alpha=alpha_now
                    )
                else:
                    print(f"[Merge] Step {step}: alpha reached 0, skipping merging.")
            else:
                print(f"[Merge] Step {step}: not a merging step, skipping.")

            buf = io.BytesIO()
            torch.save(resulted_state_dicts, buf)
            buf.seek(0)
            self.aggregated[step] = buf.read()
            del resulted_state_dicts  # 释放聚合结果
            buf.close()               # 关闭buffer

            print(f"[Store] Aggregated weights for at step {step} stored.")


        print(f"[Aggregate] Aggregation finished for step {step}. Aggregated weights ready for download.")
        del self.uploads[step]

        # Only keep the most recent `keep_steps` aggregated weights
        for old_step in list(self.aggregated.keys()):
            if old_step < step - self.keep_steps:
                print(f"[Cleanup] Removing old aggregated weights for step {old_step}")
                del self.aggregated[old_step]


import torch
import torch.nn.functional as F
from collections import defaultdict

def merge_experts_task_vector_topk_cosine_w1_per_layer(resulted_state_dicts, topk=4, alpha=0.05):
    """
    多层 MoE expert 融合版本（余弦相似度版本）：
    - 按层分别计算相似度
    - 相似度只用 w1
    - 融合作用于该 expert 的所有参数
    """
    # 按 layer 收集参数
    layer_expert_params = defaultdict(dict)  # {layer_id: {expert_id: {param_name: tensor}}}

    for key, val in resulted_state_dicts.items():
        if "ffn.experts.mlp" in key:
            parts = key.split(".")
            layer_id = int(parts[2])  # 假设 key 是 layers.{layer}.ffn.experts.mlp...
            if "expert_w1" in key or "expert_v1" in key or "expert_w2" in key:
                expert_id = int(parts[-1])
                if expert_id not in layer_expert_params[layer_id]:
                    layer_expert_params[layer_id][expert_id] = {}
                layer_expert_params[layer_id][expert_id][key] = val

    # 遍历每一层
    for layer_id, experts in layer_expert_params.items():
        expert_ids = sorted(experts.keys())

        # 1. 收集 w1 向量
        w1_vecs = []
        for eid in expert_ids:
            for k, v in experts[eid].items():
                if "expert_w1" in k:
                    w1_vecs.append(v.reshape(-1))
                    break  # 每个 expert 只有一个 w1

        w1_matrix = torch.stack(w1_vecs)  # [num_experts, dim]

        # 2. 相似度矩阵（使用 cosine similarity）
        sim_matrix = F.cosine_similarity(
            w1_matrix.unsqueeze(1),  # [num_experts, 1, dim]
            w1_matrix.unsqueeze(0),  # [1, num_experts, dim]
            dim=2
        )  # [num_experts, num_experts]

        # 3. 遍历每个 expert
        for i, eid in enumerate(expert_ids):
            # 排除自己
            sim_scores = sim_matrix[i]
            sim_scores[i] = float('-inf')

            # top-K
            topk_idx = torch.topk(sim_scores, k=topk, largest=True).indices.tolist()
            topk_eids = [expert_ids[j] for j in topk_idx]

            # base expert 向量（全参数）
            base_vec = torch.cat([p.reshape(-1) for p in experts[eid].values()])

            # 平均 task vector
            task_vectors = []
            for donor_eid in topk_eids:
                donor_vec = torch.cat([p.reshape(-1) for p in experts[donor_eid].values()])
                task_vectors.append(donor_vec - base_vec)
            avg_task_vector = torch.stack(task_vectors).mean(dim=0)

            # 融合
            merged_vec = base_vec + alpha * avg_task_vector

            # 更新回 state_dict
            offset = 0
            for pkey, pval in experts[eid].items():
                numel = pval.numel()
                new_tensor = merged_vec[offset:offset + numel].view_as(pval)
                resulted_state_dicts[pkey] = new_tensor
                offset += numel

    return resulted_state_dicts

def serve(total_peers, port=50051, num_train_experts_per_node=1,
          merge_interval=200, merge_alpha_start=0.1, merge_decay_steps=4000):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20),
                         options=[
                             ('grpc.max_send_message_length', 1937*1024*1024),
                             ('grpc.max_receive_message_length', 1937*1024*1024),
                             ('grpc.max_metadata_size', 120 * 1024)
                         ])
    federated_pb2_grpc.add_FederatedServerServicer_to_server(
        FederatedServer(total_peers,
                        num_train_experts_per_node=num_train_experts_per_node,
                        merge_interval=merge_interval,
                        merge_alpha_start=merge_alpha_start,
                        merge_decay_steps=merge_decay_steps),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    print(f"[Server] Server started on port {port} for {total_peers} peers.")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_peers', type=int, default=2, help="总的client数")
    parser.add_argument('--num_train_experts_per_node', type=int, default=8)
    parser.add_argument('--port', type=int, default=50051)
    parser.add_argument('--merge_interval', type=int, default=500)
    parser.add_argument('--merge_alpha_start', type=float, default=0.01)
    parser.add_argument('--merge_decay_steps', type=int, default=10000)
    args = parser.parse_args()

    serve(args.total_peers, args.port, args.num_train_experts_per_node,
          args.merge_interval, args.merge_alpha_start, args.merge_decay_steps)
