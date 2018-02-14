#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/ga_sync_gpu.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

#include "armci.h"
#include "comex.h"

namespace caffe {

template<typename Dtype>
static void get_pointers(const vector<Blob<Dtype>*>& blobs,
    Dtype *ptr, vector<Dtype*>& ptrs)
{
  for (int i = 0; i < blobs.size(); ++i) {
    ptrs[i] = ptr;
    ptr += blobs[i]->count();
  }
}


static int global_device_id;
static cudaStream_t global_stream;

static void* alloc(size_t size)
{
  void *memory = NULL;
  GPUMemory::allocate(&memory, size, global_device_id, global_stream);
  return memory;
}

static void dealloc(void *memory)
{
}

template<typename Dtype>
GASyncGPU<Dtype>::GASyncGPU(shared_ptr<Solver<Dtype> > root_solver)
    : stream_(),
      comm_rank_(),
      comm_size_(),
      solver_(),
      sgdsolver_(),
      params_(root_solver->net()->learnable_params()),
      time_comm_(),
      time_comp_(),
      stats_comm_(),
      stats_comp_(),
      data_recv_(),
      hist_recv_(),
      data_hdl_(),
      hist_hdl_(),
      first_time_(true)
{
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  CUDA_CHECK(cudaGetDeviceCount(&count));

  comm_size_ = caffe::mpi::comm_size();
  node_rank = caffe::mpi::node_rank();
  node_size = caffe::mpi::node_size();

  if (node_size <= count) {
    if (count != node_size) {
      LOG(INFO) << "GASyncGPU MPI node size < cudaGetDeviceCount";
    }
    CUDA_CHECK(cudaSetDevice(node_rank));
  }
  else {
    throw std::runtime_error("GASyncGPU too many MPI ranks per node");
  }

  if (0 == node_rank ) {
    cudaDeviceProp device_prop;
    for (int i = 0; i < count; ++i) {
      cudaGetDeviceProperties(&device_prop, i);
      LOG(INFO) << "GPU " << i << ": " << device_prop.name;
    }
  }

  solver_ = root_solver;
  solver_->add_callback(this);
  solver_->set_use_mpi(true);

  stats_clear(&stats_comm_);
  stats_clear(&stats_comp_);

  sgdsolver_ = boost::dynamic_pointer_cast<SGDSolver<Dtype> >(root_solver);
  if (NULL == sgdsolver_) {
    LOG(FATAL) << "dynamic cast of SGDSolver failed";
  }

  stream_ = GPUMemory::device_stream(node_rank);
  global_stream = stream_;
  global_device_id = node_rank;
  comex_malloc_local_set_custom(alloc, dealloc);

  data_pointers_.resize(params_.size());
  for (size_t i=0; i<params_.size(); ++i) {
    data_pointers_[i].resize(comm_size_);
    /* allocate memory */
    ARMCI_Malloc(reinterpret_cast<void**>(&data_pointers_[i][0]),
        sizeof(Dtype)*params_[i]->count());
    /* init memory to current value of param */
    caffe_copy(params_[i]->count(),
        reinterpret_cast<const Dtype*>(params_[i]->data()->cpu_data()),
        data_pointers_[i][comm_rank_]);
    /* replace param pointer */
    params_[i]->data()->set_gpu_data(data_pointers_[i][comm_rank_]);
  }

  /* allocate ARMCI buffers for model history */
  hist_pointers_.resize(params_.size());
  for (size_t i=0; i<params_.size(); ++i) {
    hist_pointers_[i].resize(comm_size_);
    /* allocate memory */
    ARMCI_Malloc(reinterpret_cast<void**>(&hist_pointers_[i][0]),
        sizeof(Dtype)*params_[i]->count());
    /* init memory to 0 */
    caffe_set(sgdsolver_->history()[i]->count(),
        Dtype(0), hist_pointers_[i][comm_rank_]);
    /* replace hist pointer */
    sgdsolver_->history()[i]->data()->set_gpu_data(hist_pointers_[i][comm_rank_]);
  }

  /* allocate local receive buffers */
  data_recv_.resize(params_.size());
  hist_recv_.resize(params_.size());
  for (size_t i=0; i<params_.size(); ++i) {
    data_recv_[i] = reinterpret_cast<Dtype*>(ARMCI_Malloc_local(sizeof(Dtype)*params_[i]->count()));
    hist_recv_[i] = reinterpret_cast<Dtype*>(ARMCI_Malloc_local(sizeof(Dtype)*params_[i]->count()));
  }

  data_hdl_.resize(params_.size());
  hist_hdl_.resize(params_.size());
}

template<typename Dtype>
GASyncGPU<Dtype>::~GASyncGPU() {
}

template<typename Dtype>
void GASyncGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void GASyncGPU<Dtype>::on_begin() {
  DLOG(INFO) << "on_begin()";
}

template<typename Dtype>
void GASyncGPU<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
  first_time_ = false;
}

template<typename Dtype>
void GASyncGPU<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(" << param_id << ")";
}

template<typename Dtype>
int GASyncGPU<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(" << param_id << ")";
  return param_id;
}

template<typename Dtype>
void GASyncGPU<Dtype>::on_forward(int param_id) {
  int victim = rand() % comm_size_;
  DLOG(INFO) << "on_forward(" << param_id << ") victim=" << victim;

#if 0
  if (!first_time_) {
    ARMCI_Wait(&data_hdl_[param_id]);
    ARMCI_Wait(&hist_hdl_[param_id]);
    /* blend with local, this also copies it to cpu from prv */
    caffe_cpu_axpby(params_[param_id]->count(),
        Dtype(0.5), data_recv_[param_id],
        Dtype(0.5), params_[param_id]->mutable_cpu_data());
    caffe_cpu_axpby(params_[param_id]->count(),
        Dtype(0.5), hist_recv_[param_id],
        Dtype(0.5), sgdsolver_->history()[param_id]->mutable_cpu_data());
  }

  /* prefetch data and history of random victim */
  ARMCI_NbGet(data_pointers_[param_id][victim],
      data_recv_[param_id],
      sizeof(Dtype)*params_[param_id]->count(),
      victim,
      &data_hdl_[param_id]);
  ARMCI_NbGet(hist_pointers_[param_id][victim],
      hist_recv_[param_id],
      sizeof(Dtype)*params_[param_id]->count(),
      victim,
      &hist_hdl_[param_id]);
#else
  /* blocking fetch data and history of random victim */
  ARMCI_Get(data_pointers_[param_id][victim],
      data_recv_[param_id],
      sizeof(Dtype)*params_[param_id]->count(),
      victim);
  ARMCI_Get(hist_pointers_[param_id][victim],
      hist_recv_[param_id],
      sizeof(Dtype)*params_[param_id]->count(),
      victim);
  /* blend with local */
  caffe_gpu_axpby(params_[param_id]->count(),
      Dtype(0.5), data_recv_[param_id],
      Dtype(0.5), params_[param_id]->mutable_gpu_data());
  caffe_gpu_axpby(params_[param_id]->count(),
      Dtype(0.5), hist_recv_[param_id],
      Dtype(0.5), sgdsolver_->history()[param_id]->mutable_gpu_data());
#endif
}

template<typename Dtype>
void GASyncGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void GASyncGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(GASyncGPU);

}  // namespace caffe

