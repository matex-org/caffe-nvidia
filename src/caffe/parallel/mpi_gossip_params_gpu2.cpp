#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/mpi_gossip_params_gpu2.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/gpu_memory.hpp"

namespace caffe {

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next() {
  if (cube_) {
    if (rotate_) {
      next_cube_rotate();
    }
    else {
      next_cube();
    }
  }
  else {
    if (rotate_) {
      next_diffuse_rotate();
    }
    else {
      next_diffuse();
    }
  }
  //LOG(INFO) << "rank " << comm_rank_orig_ << " rot rank " << comm_rank_ << " send " << send_pair_ << " recv " << recv_pair_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next_cube() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next_cube_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    mci_ = (mci_+1)%comm_size_;
    comm_rank_ = caffe::mpi::comm_rank(comms_[mci_]);
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next_diffuse() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  recv_pair_ = comm_rank_ + int(pow(2,hci_));
  send_pair_ = comm_rank_ - int(pow(2,hci_));
  if (recv_pair_ >= comm_size_) {
    recv_pair_ = recv_pair_ - comm_size_;
  }
  if (send_pair_ < 0) {
    send_pair_ = send_pair_ + comm_size_;
  }
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::next_diffuse_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    mci_ = (mci_+1)%comm_size_;
    comm_rank_ = caffe::mpi::comm_rank(comms_[mci_]);
  }
  recv_pair_ = comm_rank_ + int(pow(2,hci_));
  send_pair_ = comm_rank_ - int(pow(2,hci_));
  if (recv_pair_ >= comm_size_) {
    recv_pair_ = recv_pair_ - comm_size_;
  }
  if (send_pair_ < 0) {
    send_pair_ = send_pair_ + comm_size_;
  }
  ++hci_;
}

template<typename Dtype>
MPIGossipParamsGPU2<Dtype>::MPIGossipParamsGPU2(
    shared_ptr<Solver<Dtype> > root_solver,
    const SolverParameter& param,
    bool cube,
    bool avgdata,
    bool rotate)
  : GPUParams<Dtype>(root_solver, param.device_id()),
    comm_rank_(),
    comm_size_(),
    logp_(0),
    hci_(0),
    mci_(0),
    send_pair_(0),
    recv_pair_(0),
    solver_(),
    params_(root_solver->net()->learnable_params()),
#ifdef USE_MPI
    comms_(),
#endif
    diff_all_(),
    data_all_(),
    cube_(cube),
    avgdata_(avgdata),
    rotate_(rotate)
{
#ifdef USE_MPI
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  CUDA_CHECK(cudaGetDeviceCount(&count));

  // one MPI_Comm per rank
  Timer timer_comm_create_;
  timer_comm_create_.Start();
  comm_size_ = caffe::mpi::comm_size();
  node_rank = caffe::mpi::node_rank();
  node_size = caffe::mpi::node_size();
  comms_.resize(comm_size_);
  comms_[0] = caffe::mpi::comm_dup();
  vector<int> ranks(comm_size_);
  for (int i = 0; i < comm_size_; ++i) {
    ranks[i] = i;
  }
  for (int i = 1; i < comm_size_; ++i) {
    for (int j = 0; j < comm_size_; ++j) {
      ranks[j] = (ranks[j]+1)%comm_size_;
    }
    comms_[i] = caffe::mpi::comm_create(ranks);
    LOG(INFO) << "my rank " << caffe::mpi::comm_rank(comms_[i]);
  }
  LOG(INFO) << "comm creation time " << timer_comm_create_.MilliSeconds();

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

  comm_rank_ = caffe::mpi::comm_rank(comms_[0]);
  comm_rank_orig_ = comm_rank_;
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  //diff_all_ = new Dtype[size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&diff_all_),
      size_ * sizeof(Dtype), param.device_id(), cudaStreamDefault);
  caffe_gpu_set(size_, Dtype(0), diff_all_);

  //data_all_ = new Dtype[size_];
  GPUMemory::allocate(reinterpret_cast<void **>(&data_all_),
      size_ * sizeof(Dtype), param.device_id(), cudaStreamDefault);
  caffe_gpu_set(size_, Dtype(0), data_all_);

#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIGossipParamsGPU2<Dtype>::~MPIGossipParamsGPU2() {
  delete [] diff_all_;
  delete [] data_all_;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::on_begin() {
  DLOG(INFO) << "on_begin()";
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::allreduce() {
  DLOG(INFO) << "allreduce()";
#ifdef USE_MPI

  // select next exchange partners
  next();

  // exchange data
  if (avgdata_) {
      MPI_Comm comm = comms_[mci_];
#if 0
      caffe::mpi::sendrecv(
              data_,     size_, send_pair_, 1234,
              data_all_, size_, recv_pair_, 1234, comm);
#endif
#if 1
      vector<MPI_Request> requests(2);
      caffe::mpi::irecv(requests[0], data_all_, size_, recv_pair_, 1234, comm);
      caffe::mpi::isend(requests[1], data_,     size_, send_pair_, 1234, comm);
      caffe::mpi::waitall(requests);
#endif
  }

  // exchange diff
  {
      MPI_Comm comm = comms_[mci_];
#if 0
      caffe::mpi::sendrecv(
              diff_,     size_, send_pair_, 1234,
              diff_all_, size_, recv_pair_, 1234, comm);
#endif
#if 1
      vector<MPI_Request> requests(2);
      caffe::mpi::irecv(requests[0], diff_all_, size_, recv_pair_, 2345, comm);
      caffe::mpi::isend(requests[1], diff_,     size_, send_pair_, 2345, comm);
      caffe::mpi::waitall(requests);
#endif
  }

  if (avgdata_) {
    // average pairwise exchange
    caffe_gpu_axpby(size_, Dtype(0.5), data_, Dtype(0.5), data_all_);
    // swap data pointer with reduction pointer
    Dtype *swap;
    swap = data_;
    data_ = data_all_;
    data_all_ = swap;
    apply_buffers(params_, data_, size_, replace_gpu);
  }

  {
    // average pairwise exchange
    caffe_gpu_axpby(size_, Dtype(0.5), diff_, Dtype(0.5), diff_all_);
    // swap diff pointer with reduction pointer
    Dtype *swap;
    swap = diff_;
    diff_ = diff_all_;
    diff_all_ = swap;
    apply_buffers(params_, diff_, size_, replace_gpu);
  }
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::allreduce(int param_id) {
  DLOG(INFO) << "allreduce(param_id)";
}

template<typename Dtype>
int MPIGossipParamsGPU2<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsGPU2<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsGPU2);

}  // namespace caffe

