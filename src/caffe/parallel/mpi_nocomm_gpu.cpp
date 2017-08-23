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
#include "caffe/parallel/mpi_nocomm_gpu.hpp"
#include "caffe/parallel/stats.h"

namespace caffe {

template<typename Dtype>
MPINoCommGPU<Dtype>::MPINoCommGPU(shared_ptr<Solver<Dtype> > root_solver,
        const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      comm_(),
      comm_size_(),
      solver_()
{
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
      LOG(INFO) << "MPINoCommGPU MPI node size < cudaGetDeviceCount";
    }
    CUDA_CHECK(cudaSetDevice(node_rank));
  }
  else {
    throw std::runtime_error("MPINoCommGPU too many MPI ranks per node");
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
  //solver_->set_use_mpi(true);
  //solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
}

template<typename Dtype>
MPINoCommGPU<Dtype>::~MPINoCommGPU() {
}

template<typename Dtype>
void MPINoCommGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPINoCommGPU<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
}

template<typename Dtype>
void MPINoCommGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPINoCommGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPINoCommGPU);

}  // namespace caffe

