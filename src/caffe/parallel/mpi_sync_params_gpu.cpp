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
#include "caffe/parallel/mpi_sync_params_gpu.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

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
MPISyncParamsGPU<Dtype>::MPISyncParamsGPU(
    shared_ptr<Solver<Dtype> > root_solver,
    const SolverParameter& param)
  : GPUParams<Dtype>(root_solver, param.device_id()),
    comm_size_(),
    solver_(),
    params_(root_solver->net()->learnable_params()),
#ifdef USE_MPI
    comm_(),
#endif
    diff_all_(),
    param_diffs_(),
    timer_comm_(),
    time_in_comm_(),
    time_per_param_(),
    stats_comm_()
{
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

  caffe::mpi::bcast(data_, size_, 0, comm_);

  GPUMemory::allocate(reinterpret_cast<void **>(&diff_all_),
      size_ * sizeof(Dtype), param.device_id(), cudaStreamDefault);
  caffe_gpu_set(size_, Dtype(0), diff_all_);
  param_diffs_.resize(params_.size());
  get_pointers(params_, diff_all_, param_diffs_);

  time_per_param_.resize(params_.size());
  stats_clear(&stats_comm_);
  for (int i=0; i<params_.size(); ++i) {
      stats_clear(&time_per_param_[i]);
  }

  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncParamsGPU<Dtype>::~MPISyncParamsGPU() {
  delete [] diff_all_;
}

template<typename Dtype>
void MPISyncParamsGPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPISyncParamsGPU<Dtype>::on_begin() {
  static int count = 0;
  count++;
  DLOG(INFO) << "on_begin()";
  stats_sample_value(&stats_comm_, time_in_comm_);
  LOG_EVERY_N(INFO, 20) << "time comm " << stats_comm_._mean
      << " +- " << stats_stddev(&stats_comm_);
  if (count == 20) {
      for (int i=0; i<params_.size(); ++i) {
          LOG(INFO) << "time comm param " << i << " " << time_per_param_[i]._mean << " +- " << stats_stddev(&time_per_param_[i]);
      }
      count = 0;
  }
  time_in_comm_ = 0.0;
}

template<typename Dtype>
void MPISyncParamsGPU<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
}

template<typename Dtype>
void MPISyncParamsGPU<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(param_id)";
  Blob<Dtype> *blob = params_[param_id];
  Dtype *sum = param_diffs_[param_id];
  timer_comm_.Start();
  caffe::mpi::allreduce_copy((const Dtype*)blob->gpu_diff(),
          sum, blob->count(), MPI_SUM, comm_);
  stats_sample_value(&time_per_param_[param_id], timer_comm_.MilliSeconds());
  time_in_comm_ += timer_comm_.MilliSeconds();
}

template<typename Dtype>
int MPISyncParamsGPU<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  Blob<Dtype> *blob = params_[param_id];
  Dtype *swap = blob->mutable_gpu_diff();
  blob->diff()->set_gpu_data(param_diffs_[param_id]);
  param_diffs_[param_id] = swap;
  return param_id;
}

template<typename Dtype>
void MPISyncParamsGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPISyncParamsGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncParamsGPU);

}  // namespace caffe

