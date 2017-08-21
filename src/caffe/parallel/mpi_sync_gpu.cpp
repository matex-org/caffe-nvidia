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
#include "caffe/parallel/stats.h"

namespace caffe {

template<typename Dtype>
MPISyncGPU<Dtype>::MPISyncGPU(shared_ptr<Solver<Dtype> > root_solver,
        const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
#ifdef USE_MPI
      comm_(),
#endif
      comm_size_(),
      solver_(),
      timer_(),
      stats_comm_(),
      cpu_ptr_()
{
#ifdef USE_MPI
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  stats_clear(&stats_comm_);

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
  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));

#ifdef NOT_COHERENT
  cpu_ptr_ = new Dtype[size_];
  // Copy from GPU device to CPU host
  CUDA_CHECK(cudaMemcpy(cpu_ptr_, data_, size_ * sizeof(Dtype), cudaMemcpyDeviceToHost));
  // bcast weights and biases
  caffe::mpi::bcast(cpu_ptr_, size_, 0, comm_);
  // Copy from CPU host to GPU device
  CUDA_CHECK(cudaMemcpy(data_, cpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyHostToDevice));
#else
  caffe::mpi::bcast(data_, size_, 0, comm_);
#endif
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncGPU<Dtype>::~MPISyncGPU() {
#ifdef NOT_COHERENT
  delete [] cpu_ptr_;
#endif
}

template<typename Dtype>
void MPISyncGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPISyncGPU<Dtype>::allreduce() {
  static int count = 0;
  count++;
  DLOG(INFO) << "allreduce()";
#ifdef USE_MPI
  // Sum gradients
  timer_.Start();
#ifdef NOT_COHERENT
  // Copy from GPU device to CPU host
  CUDA_CHECK(cudaMemcpy(cpu_ptr_, diff_, size_ * sizeof(Dtype), cudaMemcpyDeviceToHost));
  // Sum gradients
  caffe::mpi::allreduce(cpu_ptr_, size_, MPI_SUM, comm_);
  // Copy from CPU host to GPU device
  CUDA_CHECK(cudaMemcpy(diff_, cpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyHostToDevice));
#else
  caffe::mpi::allreduce(diff_, size_, MPI_SUM, comm_);
#endif
  double time_comm_ = timer_.MilliSeconds();
  stats_sample_value(&stats_comm_, time_comm_);
  if (count == 20) {
    count = 0;
    LOG(INFO) << "time comm sample " << time_comm_;
  }
  LOG_EVERY_N(INFO, 20) << "time comm " << stats_comm_._mean
    << " += " << stats_stddev(&stats_comm_)
    << " min " << stats_comm_._min
    << " max " << stats_comm_._max;
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

