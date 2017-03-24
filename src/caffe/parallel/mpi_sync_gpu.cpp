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
#include "caffe/parallel/mpi_sync_gpu.hpp"

namespace caffe {

template<typename Dtype>
MPISyncGPU<Dtype>::MPISyncGPU(shared_ptr<Solver<Dtype> > root_solver,
const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
#ifdef USE_MPI
      comm_(),
#endif
      comm_size_(),
      timer_(),
      solver_() {
#ifdef USE_MPI
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  CUDA_CHECK(cudaGetDeviceCount(&count));

  comm_ = caffe::mpi::comm_dup();
  comm_size_ = caffe::mpi::comm_size(comm_);
  node_rank = caffe::mpi::node_rank(comm_);
  node_size = caffe::mpi::node_size(comm_);

  if (node_size <= count) {
    if (count != node_size) {
      LOG(INFO) << "MPISyncGPU MPI node size < cudaGetDeviceCount";
    }
    CUDA_CHECK(cudaSetDevice(node_rank));
  }
  else {
    throw std::runtime_error("MPISyncGPU too many MPI ranks per node");
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

  caffe::mpi::bcast(data_, size_, 0, comm_);
  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncGPU<Dtype>::~MPISyncGPU() {
}

template<typename Dtype>
void MPISyncGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPISyncGPU<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
#ifdef USE_MPI
  // Sum gradients
  timer_.Start();
  caffe::mpi::allreduce(diff_, size_, MPI_SUM, comm_);
  double time_comm_ = timer_.MilliSeconds();
  DLOG(INFO) << "time comm " << time_comm_;
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPISyncGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPISyncGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncGPU);

}  // namespace caffe

