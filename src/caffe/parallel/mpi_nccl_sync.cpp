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
#include "caffe/parallel/mpi_nccl_sync.hpp"
#include "caffe/util/blocking_queue.hpp"

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
MPINCCLSync<Dtype>::MPINCCLSync(shared_ptr<Solver<Dtype> > root_solver,
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
      solver_() {
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
      LOG(INFO) << "MPINCCLSync MPI node size < cudaGetDeviceCount";
    }
    CUDA_CHECK(cudaSetDevice(node_rank));
  }
  else {
    throw std::runtime_error("MPINCCLSync too many MPI ranks per node");
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

#else
  NO_GPU;
#endif
#else
  NO_MPI;
#endif
#endif
}

template<typename Dtype>
MPINCCLSync<Dtype>::~MPINCCLSync() {
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));
  CUDA_CHECK(cudaSetDevice(initial_device));
  ncclCommDestroy(ncclComm_);
#endif
#endif
#endif
}

template<typename Dtype>
void MPINCCLSync<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPINCCLSync<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
#ifdef USE_NCCL
#ifdef USE_MPI
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif
  Timer timer;
  timer.Start();
  NCCLCHECK(ncclAllReduce((const void *)diff_, (void*)diff_, size_,
        nccl::dataType<Dtype>::type, ncclSum, ncclComm_, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  LOG(INFO) << "time in allreduce " << timer.MilliSeconds();
#else
  NO_GPU;
#endif
#endif
#endif
}

template<typename Dtype>
void MPINCCLSync<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPINCCLSync<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPINCCLSync);

}  // namespace caffe

