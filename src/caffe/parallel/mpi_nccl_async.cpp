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
#include "caffe/parallel.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/mpi_nccl_async.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/gpu_memory.hpp"

#ifdef USE_NCCL
#include "nccl.h"

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("NCCL failure %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#endif

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

template<typename Dtype>
static void set_pointers(const vector<Blob<Dtype>*>& blobs, vector<Dtype*>& ptrs)
{
  for (int i = 0; i < blobs.size(); ++i) {
    blobs[i]->diff()->set_gpu_data(ptrs[i]);
  }
}

template<typename Dtype>
MPINCCLAsync<Dtype>::MPINCCLAsync(shared_ptr<Solver<Dtype> > root_solver,
                        const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
#ifdef USE_MPI
      comm_(),
      comm_size_(),
#endif
#ifdef USE_NCCL
      ncclComm_(),
      stream_(),
#endif
      solver_(),
      params_(root_solver->net()->learnable_params()),
    diff_all_(),
    param_diffs_()
{
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
  int count = 0;
  int node_rank = 0;
  int node_size = 0;
  ncclUniqueId commId;
  ncclResult_t ret;

  CUDA_CHECK(cudaGetDeviceCount(&count));

  comm_ = caffe::mpi::comm_dup();
  comm_size_ = caffe::mpi::comm_size(comm_);
  node_rank = caffe::mpi::node_rank(comm_);
  node_size = caffe::mpi::node_size(comm_);

  if (node_size <= count) {
    if (count != node_size) {
      LOG(INFO) << "MPINCCLAsync MPI node size < cudaGetDeviceCount";
    }
    CUDA_CHECK(cudaSetDevice(node_rank));
  }
  else {
    throw std::runtime_error("MPINCCLAsync too many MPI ranks per node");
  }

  if (0 == node_rank ) {
    cudaDeviceProp device_prop;
    for (int i = 0; i < count; ++i) {
      cudaGetDeviceProperties(&device_prop, i);
      LOG(INFO) << "GPU " << i << ": " << device_prop.name;
    }

  }

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);
  solver_->set_use_mpi(true);

  NCCLCHECK(ncclGetUniqueId(&commId));
  MPI_Bcast(&commId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, comm_);
  ret = ncclCommInitRank(&ncclComm_, node_size, commId, node_rank);
  if (ret != ncclSuccess) {
    printf("NCCL Init failed (%d) '%s'\n", ret, ncclGetErrorString(ret));
    exit(1);
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

  NCCLCHECK(ncclBcast((void*)data_, size_, nccl::dataType<Dtype>::type,
        0, ncclComm_, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));

  GPUMemory::allocate(reinterpret_cast<void **>(&diff_all_),
      size_ * sizeof(Dtype), param.device_id(), stream_);
  caffe_gpu_set(size_, Dtype(0), diff_all_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

  param_diffs_.resize(params_.size());
  get_pointers(params_, diff_all_, param_diffs_);
#else
  NO_GPU;
#endif
#else
  NO_MPI;
#endif
#endif
}

template<typename Dtype>
MPINCCLAsync<Dtype>::~MPINCCLAsync() {
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));
  GPUMemory::deallocate(diff_all_, self, stream_);
  CUDA_CHECK(cudaSetDevice(initial_device));
  ncclCommDestroy(ncclComm_);
#endif
#endif
#endif
}

template<typename Dtype>
void MPINCCLAsync<Dtype>::soft_barrier() {
#ifndef CPU_ONLY
  // CPU barrier to avoid busy-polling on the GPU.
  //MPI_Barrier(comm_);
#endif
}

template<typename Dtype>
void MPINCCLAsync<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  //MPI_Barrier(comm_);
#endif
#endif
#endif
}

template<typename Dtype>
void MPINCCLAsync<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  /* swap pointers */
#if 1
  Dtype *swap = diff_;
  diff_ = diff_all_;
  diff_all_ = swap;
  set_pointers(params_, param_diffs_);
  get_pointers(params_, diff_all_, param_diffs_);
#endif
#else
  NO_GPU;
#endif
#endif
#endif
}

template<typename Dtype>
void MPINCCLAsync<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(int param_id)";
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
  Blob<Dtype> *blob = params_[param_id];
#if 1
  NCCLCHECK(ncclAllReduce(
        (const void *)blob->gpu_diff(),
        (void*)param_diffs_[param_id],
        blob->count(),
        nccl::dataType<Dtype>::type,
        ncclSum,
        ncclComm_,
        stream_));
#else
  NCCLCHECK(ncclAllReduce(
        (const void *)blob->gpu_diff(),
        (void*)blob->mutable_gpu_diff(),
        blob->count(),
        nccl::dataType<Dtype>::type,
        ncclSum,
        ncclComm_,
        stream_));
#endif
#else
  NO_GPU;
#endif
#endif
#endif
}

template<typename Dtype>
void MPINCCLAsync<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPINCCLAsync<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPINCCLAsync);

}  // namespace caffe

