#ifndef CAFFE_PARALLEL_MPI_SYNC_PARAMS_GPU_HPP_
#define CAFFE_PARALLEL_MPI_SYNC_PARAMS_GPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/solver.hpp"

namespace caffe {

// Synchronous data parallelism using Allreduce between remote GPUs.
template<typename Dtype>
class MPISyncParamsGPU : public GPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPISyncParamsGPU(shared_ptr<Solver<Dtype> > root_solver,
          const SolverParameter& param);
  virtual ~MPISyncParamsGPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:
  void on_start();
  void on_begin();
  void allreduce();
  void allreduce(int param_id);
  int on_apply(int param_id);

  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;
  const vector<Blob<Dtype>*>& params_;
#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  Dtype *diff_all_;
  vector<Dtype*> param_diffs_;
  Timer timer_comm_;
  double time_in_comm_;
  vector<stats_t> time_per_param_;
  stats_t stats_comm_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

