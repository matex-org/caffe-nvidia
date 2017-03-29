#ifndef CAFFE_PARALLEL_MPI_SYNC_GPU_HPP_
#define CAFFE_PARALLEL_MPI_SYNC_GPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Synchronous data parallelism using Allreduce between remote GPUs.
template<typename Dtype>
class MPISyncGPU : public GPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPISyncGPU(shared_ptr<Solver<Dtype> > root_solver,
          const SolverParameter& param,
          bool coherent);
  virtual ~MPISyncGPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:
  void on_start();
  void allreduce();

#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;
  Timer timer_;
  bool coherent_;

  Dtype* cpu_ptr_;
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
