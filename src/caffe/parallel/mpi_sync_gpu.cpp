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
MPISyncGPU<Dtype>::MPISyncGPU(shared_ptr<Solver<Dtype> > root_solver)
    : GPUParams<Dtype>(root_solver, root_solver->param().device_id()),
#ifdef USE_MPI
      comm_(),
#endif
      comm_size_(),
      solver_(),
      timer_(),
      cpu_ptr_() {
#ifdef USE_MPI
  #if CAFFE_FT
  comm_ = caffe::mpi::get_working_comm();
  std::cout << "Working Comm MPISYNCCPU.\n";
  #else
  comm_ = caffe::mpi::comm_dup();
  #endif /*CAFFE_FT*/
  comm_size_ = caffe::mpi::comm_size(comm_);

  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  CUDA_CHECK(cudaGetDeviceCount(&count));

  // comm_ = caffe::mpi::comm_dup();
  comm_size_ = caffe::mpi::comm_size(comm_);
  node_rank = caffe::mpi::node_rank(comm_);
  node_size = caffe::mpi::node_size(comm_);

  if (node_size <= count) {
    if (count != node_size) {
      LOG(INFO) << "MPI node size < cudaGetDeviceCount";
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
#ifdef CAFFE_FT
  caffe::mpi::bcast(data_, size_, 0, comm_);
  LOG(INFO) << "My rank after bcast: " << caffe::mpi::comm_rank(caffe::mpi::get_working_comm());
#endif /*CAFFE_FT*/
  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));

  cpu_ptr_ = new Dtype[size_];

  // Copy from GPU device to CPU host
  CUDA_CHECK(cudaMemcpy(cpu_ptr_, data_, size_ * sizeof(Dtype), cudaMemcpyDeviceToHost));

  // bcast weights and biases
  caffe::mpi::bcast(cpu_ptr_, size_, 0, comm_);

  // Copy from CPU host to GPU device
  CUDA_CHECK(cudaMemcpy(data_, cpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyHostToDevice));

#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncGPU<Dtype>::~MPISyncGPU() {
  delete [] cpu_ptr_;
}

template<typename Dtype>
#ifdef CAFFE_FT
std::tuple<int, bool> MPISyncGPU<Dtype>::allreduce() {
#else
void MPISyncGPU<Dtype>::allreduce() {
#endif
  DLOG(INFO) << "allreduce()";
#ifndef CPU_ONLY
  // Copy from GPU device to CPU host
  CUDA_CHECK(cudaMemcpy(cpu_ptr_, diff_, size_ * sizeof(Dtype), cudaMemcpyDeviceToHost));

  // Sum gradients
#ifdef CAFFE_FT
  comm_ = caffe::mpi::get_working_comm();
  std::tuple<int,bool> ret_val
      = caffe::mpi::allreduce(cpu_ptr_, size_, MPI_SUM, this->comm_);
  if(std::get<1>(ret_val)) {
    this->comm_ = caffe::mpi::get_working_comm();
    DLOG(INFO) << "RETVAL<1> true, MPISYNCGPU --------------" ;
  }
  if(std::get<0>(ret_val) != MPI_SUCCESS) { // This should not be triggered
    comm_ = caffe::mpi::get_working_comm();
    int temp_sz = caffe::mpi::comm_size(comm_);
    DLOG(INFO) << "Corrected Communicator Size {mpi_sync_cpu}!!!!!: " << temp_sz;
  }
  return ret_val;
#else
  caffe::mpi::allreduce(cpu_ptr_, size_, MPI_SUM, comm_);
#endif

  // Copy from CPU host to GPU device
  CUDA_CHECK(cudaMemcpy(diff_, cpu_ptr_, size_ * sizeof(Dtype), cudaMemcpyHostToDevice));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
void MPISyncGPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

#ifdef CAFFE_FT
#ifdef SNAPSHOT_RESTART
template<typename Dtype>
void MPISyncGPU<Dtype>::Run(const string snapshot_file) {
  LOG(INFO) << "Restarting Optimization from Snapshot File";
  // Re run the solver on current thread
  solver_->Solve(snapshot_file.c_str());
}
#endif
#endif

template<typename Dtype>
void MPISyncGPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncGPU);

}  // namespace caffe
