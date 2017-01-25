#ifndef CAFFE_PARALLEL_MPI_NCCL_SYNC_HPP_
#define CAFFE_PARALLEL_MPI_NCCL_SYNC_HPP_

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"

#ifdef USE_NCCL
#include "nccl.h"
#endif

namespace caffe {

// Synchronous data parallelism using Allreduce between local GPUs.
template<typename Dtype>
class MPINCCLSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPINCCLSync(shared_ptr<Solver<Dtype> > root_solver,
                   const SolverParameter& param);
  virtual ~MPINCCLSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

  void allreduce(int param_id) {}
  void syncCommStream() {}

 protected:
  void on_start();
  void allreduce();
  void soft_barrier() {}

#ifdef USE_MPI
  MPI_Comm comm_;
  int comm_size_;
#endif
#ifdef USE_NCCL
  ncclComm_t ncclComm_;
  cudaStream_t stream_;
#endif
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

