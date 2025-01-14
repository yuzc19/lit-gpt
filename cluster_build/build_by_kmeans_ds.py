import faiss
from faiss import write_index, read_index
import time 
import pickle
import numpy as np
import fileinput
import sys
import argparse
import os
from datasets import load_from_disk
import datasets

def read_embed_shape(dataset, column_name="__embedding"):
    nvecs = len(dataset)
    dim = len(dataset[0][column_name])
    return nvecs, dim

def get_invlist(invlists, l): 
    """ returns the inverted lists content of l. 
    That the data is *not* copied: if the inverted index is deallocated or changes, accessing the array may crash.
    To avoid this, just clone the output arrays on output. """
    ls = invlists.list_size(l)
    list_ids = faiss.rev_swig_ptr(invlists.get_ids(l), ls)
    list_codes = faiss.rev_swig_ptr(invlists.get_codes(l), ls * invlists.code_size)
    return list_ids, list_codes

parser = argparse.ArgumentParser()
parser.add_argument("--dest_dir")
parser.add_argument("--dataset_path")
parser.add_argument("--nlist")
parser.add_argument("--metrics", default='dot')
args = parser.parse_args()

# print configs
print(f'--- --- \n building from dataset {args.dataset_path} \n \
    into {args.nlist} clusters \n \
    to destination at {args.dest_dir} \n --- ---')

n_clusters = int(args.nlist)
dataset_path = args.dataset_path

# Load dataset
dataset = datasets.concatenate_datasets([load_from_disk(f"{dataset_path}/{i}") for i in range(8)])
print(dataset)

vector_num, vector_dim = read_embed_shape(dataset)
vector_dim = int(vector_dim)  # Ensure vector_dim is an integer

print(type(vector_num), type(vector_dim))
print(f"number of vectors: {vector_num}, dimension: {vector_dim}")

# Ensure there are enough vectors to train the index
if vector_num < n_clusters:
    raise ValueError(f"Number of vectors ({vector_num}) is less than the number of clusters ({n_clusters}).")

print("----------------Index-----------")
if args.metrics == "dot":
    metrics = faiss.METRIC_INNER_PRODUCT
    coarse_quantizer = faiss.IndexFlatIP(vector_dim)
else: 
    metrics = faiss.METRIC_L2 
    coarse_quantizer = faiss.IndexFlatL2(vector_dim)
index = faiss.IndexIVFFlat(coarse_quantizer, vector_dim, n_clusters, metrics)

# # process info 
dest_dir = args.dest_dir
idx_dir = os.path.join(dest_dir, 'idx')
if not os.path.exists(idx_dir):  
    os.makedirs(idx_dir)
    
# ########## LOADING VECTORS 
print("------------------- Load Embeddings -----------------")

print("building index in shards...")
start = 0 
end = vector_num
chunk_size = 500000
while start < end: 
    shard = min(chunk_size, end - start)
    subset = dataset.select(range(start, start + shard))
    embeds = np.array(subset["__embedding"], dtype=np.float32)
    # embeds = np.array([dataset[i]["__embedding"] for i in range(start, start + shard)], dtype=np.float32)
    index.train(embeds)
    index.add(embeds) # dataset
    print(f"embedding the {start}th to {start+shard}th vectors...")
    sys.stdout.flush()
    start += shard
    
# write index
print("writing index...")
output_name = f'IVFFlat_{n_clusters}_{args.metrics}.bin'
write_index(index, os.path.join(idx_dir, output_name))
    
# store the inv lists
invlists = index.invlists

print("checking which cluster the vectors belong to...")
start = 0 
end = vector_num
while start < end: 
    shard = min(chunk_size, end - start)
    subset = dataset.select(range(start, start + shard))
    embeds = np.array(subset["__embedding"], dtype=np.float32)
    # embeds = np.array([dataset[i]["__embedding"] for i in range(start, start + shard)], dtype=np.float32)
    if start == 0:
        c_assignments = coarse_quantizer.assign(embeds, 1)  # n*1
    else: 
        c_assignments = np.concatenate((c_assignments, coarse_quantizer.assign(embeds, 1)))
    start += shard

print("writing assignments...")
centroids = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), n_clusters * vector_dim)
centroids = centroids.reshape(n_clusters, vector_dim)

# output kmeans result
center_file = os.path.join(dest_dir, 'centroid.pkl')
assign_file = os.path.join(dest_dir, 'assignment.pkl')

print(f"writing centroids to {center_file}...")
with open(center_file, 'wb') as f:
    pickle.dump(centroids, f)

print(f"writing assignments to {assign_file}...")
with open(assign_file, 'wb') as f:
    pickle.dump(c_assignments, f)