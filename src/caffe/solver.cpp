#include <algorithm>
#include <map>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <numeric>

#include <unistd.h>

// #include "boost/bind.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false),
      scale_on_apply_(1.0),
#ifdef SNAPSHOT_RESTART
      reinit_time_(0),
      snapshot_time_(0),
#endif
      iteration_timer_(), iterations_last_() {
  Init(param);
#ifdef CAFFE_FT
  const char* env_victim_rank = std::getenv("ENV_VICTIM_RANK");
  victim_rank_ = (env_victim_rank != NULL) ? atoi (env_victim_rank):-1;
  snapshot_count_ = 0;
  restart_from_snapshot_ = false;
  snapshot_model_filename_ = "";
  snapshot_solver_filename_ = "";
#endif /*CAFFE_FT*/
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false),
      scale_on_apply_(1.0),
#ifdef SNAPSHOT_RESTART
      reinit_time_(0),
      snapshot_time_(0),
#endif
      iteration_timer_(), iterations_last_() {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
#ifdef CAFFE_FT
  const char* env_victim_rank = std::getenv("ENV_VICTIM_RANK");
  victim_rank_ = (env_victim_rank != NULL) ? atoi (env_victim_rank):-1;
  snapshot_count_ = 0;
  restart_from_snapshot_ = false;
  snapshot_model_filename_ = "";
#endif /*CAFFE_FT*/
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  #ifdef USE_MPI
  #ifdef CAFFE_FT
  MPI_Comm temp_comm = caffe::mpi::get_working_comm();
  ft_rank = caffe::mpi::comm_rank(temp_comm);
  ft_size = caffe::mpi::comm_size(temp_comm);
  #else
  ft_rank = caffe::mpi::comm_rank(MPI_COMM_WORLD);
  ft_size = caffe::mpi::comm_size(MPI_COMM_WORLD);
  #endif
  #endif

  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  iteration_timer_.Start();

  double temp_time = 0;

#ifdef CAFFE_FT
  MPI_Comm test_comm = caffe::mpi::get_working_comm();
  int original_rank = caffe::mpi::comm_rank(test_comm);
#endif
  for (int i = 0; i < callbacks_.size(); ++i) {
    // we need to sync all threads before starting, otherwise some cuda init,
    // malloc or other cuda stuff could interlock with in-loop cuda GPU sync
    // called in on_start.
    callbacks_[i]->soft_barrier();
    // Initial bcast of parameters
    callbacks_[i]->on_start();
  }
  Timer iter_timer;
  double total_time = 0, total_comm_time = 0;
  net_->SetSolver(this);

  while (iter_ < stop_iter) {
  double total_step_time = 0
          , iter_time = 0
          , comm_step_time = 0
          , temp_time = 0
          , data_re_readtime = 0
          , grad_update_time = 0
          , comp_step_time = 0;

#ifdef CAFFE_FT
    MPI_Comm temp_comm = caffe::mpi::get_working_comm();
    ft_rank = caffe::mpi::comm_rank(temp_comm);
    ft_size = caffe::mpi::comm_size(temp_comm);
    // Fault Injection
    // int victim = ft_size - 1;

    if((ft_rank == victim_rank_) && (iter_ == 300)) {
      LOG(INFO) << "Victim Rank: " << victim_rank_ << std::endl;
      raise(SIGKILL);
    }
#endif
    iter_timer.Start();
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      if (Caffe::root_solver()) {
        TestAll();
      }
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
      for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->soft_barrier();
      }
    }
    temp_time = iter_timer.MilliSeconds();
    // total_step_time += temp_time;
    comm_step_time += temp_time;

    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    iter_timer.Start();
    // accumulate the loss and gradient
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();
    temp_time = iter_timer.MilliSeconds();
    comp_step_time += temp_time;
    // iter_time += temp_time;
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() <<" iter), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
#ifndef CPU_ONLY
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
#endif
    std::tuple<int, bool> ret_val;

    iter_timer.Start();
    for (int i = 0; i < callbacks_.size(); ++i) {
#ifdef CAFFE_FT
      ret_val = callbacks_[i]->allreduce();
	  temp_time = iter_timer.MilliSeconds();
	  comm_step_time += temp_time;
	  
	  iter_timer.Start();
#ifndef SNAPSHOT_RESTART
      if(std::get<1>(ret_val)) {
        // readback_timer_.Start();
        // fault has occured
        // MPI AllReduce other ranks as well..
        // Global Faulted Variable... (to trigger read from every rank
        net_->ReSetUpLayer("data");
        // temp_data_readtime =  iter_timer.MilliSeconds();

        MPI_Comm temp_comm = caffe::mpi::get_working_comm();
        ft_rank = caffe::mpi::comm_rank(temp_comm);
        ft_size = caffe::mpi::comm_size(temp_comm);
        temp_time = iter_timer.MilliSeconds();
        data_re_readtime += temp_time;
        DLOG(INFO) << "ReSetUpLayer Done:--------------rank:" << ft_rank
          << " ,size:" << ft_size << " DataReadback Time: "
          << data_re_readtime;
        iter_timer.Start();
      }
#endif /* SNAPSHOT_RESTART */
#else	  
      callbacks_[i]->allreduce();
	  // temp_time = iter_timer.MilliSeconds();
#endif 
    }
    // Make sure all gradient exchanges have finished in per-level scheme
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->syncCommStream();
    }
#ifdef CAFFE_FT
#ifdef SNAPSHOT_RESTART
    if(std::get<1>(ret_val)) {
      LOG(INFO) << "Fault Detected, Restart Initiate!";
      restart_from_snapshot_ = true;
      requested_early_exit_ = true; // Due to fault
#ifdef USE_REINIT
      if((param_.snapshot() && Caffe::root_solver())) {
        snapshot_timer_.Start();
        Snapshot();
        // Take Snapshot Timing Only Once for ReInit
        this->snapshot_time_ = snapshot_timer_.MilliSeconds();
        ++(this->snapshot_count_);
        LOG(INFO) << "Snapshotting Time(REINIT): " << this->snapshot_time_;
      }
      else
        LOG(INFO) << "Snapshotting Not done for ReINIT, Snapshot Filename missing? ";
#endif
      break;
    }
#endif /* SNAPSHOT_RESTART */
#endif /*CAFE_FT*/

    iter_timer.Start();
    ApplyUpdate();
	grad_update_time = iter_timer.MilliSeconds();
	// iter_time += iter_timer.MilliSeconds();

    total_step_time
      = comp_step_time + grad_update_time + data_re_readtime + comm_step_time ;

    total_time += total_step_time;
    total_comm_time += comm_step_time;

#ifdef USE_MPI
      LOG(INFO) << "iter " << iter_ << ", step_communication_time: " << comm_step_time << " ms with_rank " << ft_rank;
      LOG(INFO) << "iter " << iter_ << ", step_total_time: " << total_step_time << " ms with_rank " << ft_rank;
      LOG(INFO) << "iter " << iter_ << ", cumulative_communication_time: " << total_comm_time << " ms";
      LOG(INFO) << "iter " << iter_ << ", cumulative_total_time: " << total_time << " ms";
#endif

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
#ifndef USE_REINIT
#ifdef SNAPSHOT_RESTART
      snapshot_timer_.Start();
#endif
      Snapshot();
#ifdef CAFFE_FT
      // Count for restarting form snapshotted file;
#ifdef SNAPSHOT_RESTART
      this->snapshot_time_ = snapshot_timer_.MilliSeconds();
      LOG(INFO) << "Snapshotting Time(User Initiated): " << this->snapshot_time_ << " ms";
      ++(this->snapshot_count_);
#endif /*SNAPSHOT_RESTART*/
#endif /*CAFFE_FT*/
#endif /*USE_REINIT*/
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
MPI_Barrier(caffe::mpi::get_working_comm());
#ifdef CAFFE_FT
if(!requested_early_exit_)
  caffe::mpi::completed(true);

/*#ifdef SNAPSHOT_RESTART
LOG(INFO) << "Snapshot Time(MilliSeconds): " << snapshot_time_
          << " , Snapshot Count:  " << snapshot_count_;
#endif*/
#endif
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

#ifdef CAFFE_FT
#ifdef SNAPSHOT_RESTART
  snapshot_model_filename_ = model_filename;
#endif /*SNAPSHOT_RESTART*/
#endif /*CAFFE_FT*/

  // if(ft_rank == 0) {
  SnapshotSolverState(model_filename);
  MPI_Barrier(caffe::mpi::get_working_comm());
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

template <typename Dtype>
void Solver<Dtype>::ShareWeights(Solver *solver)
{
  const vector<Blob<Dtype>*>& this_net =
      this->net()->learnable_params();
  const vector<Blob<Dtype>*>& that_net =
      solver->net()->learnable_params();
  CHECK_EQ(this_net.size(), that_net.size())
      << "solvers must have identical network shapes";
  for (int i = 0; i < this_net.size(); ++i) {
    int this_size = this_net[i]->count();
    int that_size = that_net[i]->count();
    CHECK_EQ(this_size, that_size)
        << "solvers must have identical network shapes, mismatch at " << i;
#ifndef CPU_ONLY
    that_net[i]->data()->set_gpu_data(this_net[i]->data()->mutable_gpu_data());
    that_net[i]->diff()->set_gpu_data(this_net[i]->diff()->mutable_gpu_data());
#else
    that_net[i]->data()->set_cpu_data(this_net[i]->data()->mutable_cpu_data());
    that_net[i]->diff()->set_cpu_data(this_net[i]->diff()->mutable_cpu_data());
#endif
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
