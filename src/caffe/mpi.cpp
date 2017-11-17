#ifdef USE_MPI
#include <mpi.h>
#endif
#include <glog/logging.h>
#include <stdlib.h>
#include <stdexcept>
#include <string>
#include <unistd.h> // for gethostid()
#include <vector>
#include "caffe/mpi.hpp"

#ifdef CAFFE_FT
#include "caffe/util/benchmark.hpp"
#endif

namespace caffe {
namespace mpi {

#ifdef USE_MPI

MPI_Comm default_comm_ = MPI_COMM_WORLD;

#ifdef CAFFE_FT
// MPI_Comm wcomm , first_comm = MPI_COMM_WORLD; // rcomm
MPI_Comm wcomm = MPI_COMM_NULL;
MPI_Errhandler errh;
// MPI_Comm_create_errhandler(caffe::mpi::verbose_errhandler, &errh);
char err_str[MPI_MAX_ERROR_STRING] = "";
int err_strlen;
int new_size;
int old_size;
bool solver_completed = false;

// [Last Comm Size, Last Rank Failed]
int_pair_vectype size_rank_pair_vec;

int last_rank_failed = -1;

void completed(bool comp) {
  solver_completed = comp;
}

void update_faulted_processes(int rank) {
  last_rank_failed = rank;
}

#endif

MPI_Comm get_comm_default() {
  #ifdef CAFFE_FT
  return wcomm;
  #else
  return default_comm_;
  #endif
}

void set_comm_default(MPI_Comm comm) {
  if (MPI_COMM_NULL != comm) {
    default_comm_ = comm;
  }
  else {
    throw std::runtime_error("cannot assign MPI_COMM_NULL as default comm");
  }
}

void init(int *argc, char ***argv) {
#ifdef CAFFE_FT
int rank = 0, size = 0, namelen = 0;
  char name[MPI_MAX_PROCESSOR_NAME];

  int rc;
  MPI_Init(argc, argv);
  FTCommunicator ftComm;
  duplicate_comm(&wcomm, MPI_COMM_WORLD);
  MPI_Comm_size(wcomm, &size);
  MPI_Comm_rank(wcomm, &rank);
  MPI_Get_processor_name(name, &namelen);
  caffe::mpi::old_size = size;
  caffe::mpi::new_size = size;

  MPI_Comm_create_errhandler(verbose_errhandler, &errh); // global
  // MPI_Comm_set_errhandler(wcomm, MPI_ERRORS_RETURN);
  MPI_Comm_set_errhandler(wcomm, errh);

  DLOG(INFO) << "Process rank " << rank << " from number of " << size
            << " processes running on " << name;

#else

  char name[MPI_MAX_PROCESSOR_NAME];
  int requested = MPI_THREAD_SINGLE;
  int rank = 0;
  int size = 0;
  int namelen = 0;

  int rc;
  MPI_Init(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(name, &namelen);
  /*if (!initialized()) {
    int provided;
    if (MPI_SUCCESS != MPI_Init_thread(
          argc, argv, MPI_THREAD_MULTIPLE, &provided)) {

      throw std::runtime_error("MPI_Init_thread failed");
    }
  }

  if (MPI_THREAD_MULTIPLE != query_thread()) {
    throw std::runtime_error("MPI threading level must be == MPI_THREAD_MULTIPLE");
  }

  if (0 != atexit(finalize)) {
    throw std::runtime_error("atexit(caffe::mpi::finalize) failed");
  }*/
#endif // CAFFE_FT
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(name, &namelen);

  LOG(INFO) << "Process rank " << rank << " from number of " << size
            << " processes running on " << name;
}

bool initialized() {
  int flag;

  if (MPI_SUCCESS != MPI_Initialized(&flag)) {
    throw std::runtime_error("MPI_Initialized failed");
  }

  return flag;
}

void finalize() {
  #ifdef CAFFE_FT
    if(caffe::mpi::solver_completed) {
      LOG(INFO) << "Force terminate processes.\n";
      int errcode;
      if(MPI_SUCCESS != MPI_Abort(MPI_COMM_WORLD, errcode) ) {
        std::cout << "MPI_Abort error code: " << errcode << "\n";
        throw std::runtime_error("MPI_Abort failed");
      }
    }
  #endif
  if (MPI_SUCCESS != MPI_Finalize()) {
    throw std::runtime_error("MPI_Finalize failed");
  }
}

int query_thread() {
  int provided;

  if (MPI_SUCCESS != MPI_Query_thread(&provided)) {
    throw std::runtime_error("MPI_Query_thread failed");
  }

  return provided;
}

MPI_Comm comm_dup(MPI_Comm comm) {
  MPI_Comm newcomm;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_dup(comm, &newcomm)) {
    throw std::runtime_error("MPI_Comm_dup failed");
    return MPI_COMM_NULL;
  }

  return newcomm;
}

MPI_Comm comm_split(int color, int key, MPI_Comm comm) {
  MPI_Comm newcomm;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_split(comm, color, key, &newcomm)) {
    throw std::runtime_error("MPI_Comm_split failed");
    return MPI_COMM_NULL;
  }

  return newcomm;
}

MPI_Comm comm_create(MPI_Group group, MPI_Comm comm) {
  MPI_Comm newcomm;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_create(comm, group, &newcomm)) {
    throw std::runtime_error("MPI_Comm_create failed");
    return MPI_COMM_NULL;
  }

  return newcomm;
}

MPI_Comm comm_create(const std::vector<int> &incl, MPI_Comm comm) {
  MPI_Group group_old;
  MPI_Group group_new;
  MPI_Comm newcomm;
  int size;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  size = comm_size(comm);
  if (size != incl.size()) {
    throw std::runtime_error("comm_create size mismatch");
    return MPI_COMM_NULL;
  }

  if (MPI_SUCCESS != MPI_Comm_group(comm, &group_old)) {
    throw std::runtime_error("MPI_Comm_group failed");
    return MPI_COMM_NULL;
  }

#ifndef CAFFE_FT
  // if (MPI_SUCCESS != MPI_Group_incl(group_old, size, &incl[0], &group_new)) {
  //   throw std::runtime_error("MPI_Group_incl failed");
  //   return MPI_COMM_NULL;
  // }
#endif /*CAFFE_FT*/

  if (MPI_SUCCESS != MPI_Comm_create(comm, group_new, &newcomm)) {
    throw std::runtime_error("MPI_Comm_create failed");
    return MPI_COMM_NULL;
  }

  return newcomm;
}

void comm_free(MPI_Comm comm) {
  if (MPI_SUCCESS != MPI_Comm_free(&comm)) {
    throw std::runtime_error("MPI_Comm_free failed");
  }
}

int comm_rank(MPI_Comm comm) {
  int rank = 0;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_rank(comm, &rank)) {
    throw std::runtime_error("MPI_Comm_size failed");
    return 0;
  }

  return rank;
}

int comm_size(MPI_Comm comm) {
  int size = 0;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_size(comm, &size)) {
    throw std::runtime_error("MPI_Comm_size failed");
    return 0;
  }

  return size;
}

int node_rank(MPI_Comm comm) {
  int size = 0;
  int rank = 0;
  int node = 0;
  long *hostid = NULL;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  size = comm_size(comm);
  rank = comm_rank(comm);
  hostid = new long[size];
  hostid[rank] = gethostid();
  if (MPI_SUCCESS != MPI_Allgather(MPI_IN_PLACE, 0, MPI_LONG,
        hostid, 1, MPI_LONG, comm)) {
    delete [] hostid;
    throw std::runtime_error("MPI_Allgather failed");
    return 0;
  }

  for (int i=0; i<rank; ++i) {
    if (hostid[i] == hostid[rank]) {
      ++node;
    }
  }

  delete [] hostid;
  return node;
}

int node_size(MPI_Comm comm) {
  int size = 0;
  int rank = 0;
  int node = 0;
  long *hostid = NULL;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  size = comm_size(comm);
  rank = comm_rank(comm);
  hostid = new long[size];
  hostid[rank] = gethostid();
  if (MPI_SUCCESS != MPI_Allgather(MPI_IN_PLACE, 0, MPI_LONG,
        hostid, 1, MPI_LONG, comm)) {
    delete [] hostid;
    throw std::runtime_error("MPI_Allgather failed");
    return 0;
  }

  for (int i=0; i<size; ++i) {
    if (hostid[i] == hostid[rank]) {
      ++node;
    }
  }

  delete [] hostid;
  return node;
}

template <>
MPI_Datatype datatype<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype datatype<double>() {
  return MPI_DOUBLE;
}

#ifdef CAFFE_FT

void allreduce_copy(const float& sendbuf, float& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 float)");
  }
}

void allreduce_copy(const double& sendbuf, double& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 double)");
  }
}

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  int rc;
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  rc = MPI_Allreduce((void*)sendbuf, recvbuf, count, MPI_FLOAT, op, comm);
  if(rc != MPI_SUCCESS)
    // caffe::mpi::error_report(rc, &comm);

  if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy float)");
  }
}

void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  int rc;
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  rc = MPI_Allreduce((void*)sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
  if(rc != MPI_SUCCESS)
    // caffe::mpi::error_report(rc, &comm);

  if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy double)");
  }
}

std::tuple<int,bool> allreduce(float& buffer, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    DLOG(INFO) << "AllReduce (Float Ref): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  trank = caffe::mpi::comm_rank(comm);
  tsize = caffe::mpi::comm_size(comm);

  std::get<0>(ret_val)
      = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_FLOAT, op, comm);
  // if(std::get<0>(ret_val) != MPI_SUCCESS) {
  while(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    // caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    // MPIX_Comm_failure_awk(comm);
    DLOG(INFO) << "_____Communicator Failed AllReduce (Float Ref): Rank: "
      << trank << ", Size: " << tsize;
    caffe::mpi::fix_communicator(&comm);
    // test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(comm);
    tsize = caffe::mpi::comm_size(comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Float Ref): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_FLOAT, op, comm);
    //return ret_val;
  }
  return ret_val;
}

std::tuple<int,bool> allreduce(double& buffer, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Double Ref): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  trank = caffe::mpi::comm_rank(comm);
  tsize = caffe::mpi::comm_size(comm);

  std::get<0>(ret_val)
      = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_DOUBLE, op, comm);
  // if(std::get<0>(ret_val) != MPI_SUCCESS) {
  while(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    // caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    // MPIX_Comm_failure_awk(comm);
    DLOG(INFO) << "______Communicator Failed AllReduce (Double Ref): Rank: "
      << trank << ", Size: " << tsize;
    caffe::mpi::fix_communicator(&comm);
    // test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(comm);
    tsize = caffe::mpi::comm_size(comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Double Ref): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_DOUBLE, op, comm);
    // return ret_val;
  }

  return ret_val;
}

std::tuple<int, bool> allreduce(float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Float Ptr): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  trank = caffe::mpi::comm_rank(comm);
  tsize = caffe::mpi::comm_size(comm);

  std::get<0>(ret_val)
      = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_FLOAT, op, comm);

  // if(std::get<0>(ret_val) != MPI_SUCCESS) {
  while(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    // caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    // MPIX_Comm_failure_awk(comm);
    DLOG(INFO) << "_____Communicator Failed AllReduce (Float Ptr): Rank: "
      << trank << ", Size: " << tsize;
    caffe::mpi::fix_communicator(&comm);
    // test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(comm);
    tsize = caffe::mpi::comm_size(comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Float Ptr): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_FLOAT, op, comm);
    // return ret_val;
  }
  return ret_val;
}

std::tuple<int, bool> allreduce(double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Double Ptr): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  trank = caffe::mpi::comm_rank(comm);
  tsize = caffe::mpi::comm_size(comm);

  std::get<0>(ret_val)
      = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, op, comm);
  // if(std::get<0>(ret_val) != MPI_SUCCESS) {
  while(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    // caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    // MPIX_Comm_failure_awk(comm);
    DLOG(INFO) << "_____Communicator Failed AllReduce (Double Ptr): Rank: "
      << trank << ", Size: " << tsize;
    caffe::mpi::fix_communicator(&comm);
    // test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(comm);
    tsize = caffe::mpi::comm_size(comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Double Ptr): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, op, comm);

    // return ret_val;
  }
  return ret_val;
}

void bcast(int* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    // comm = get_comm_default();
    comm = get_working_comm();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_INT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

void bcast(float* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    // comm = get_comm_default();
    comm = get_working_comm();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_FLOAT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

void bcast(double* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    // comm = get_comm_default();
    comm = get_working_comm();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

int mpix_comm_replace(MPI_Comm comm, MPI_Comm* pnewcomm)
{
  MPI_Comm shrinked;
  MPI_Group cgrp, sgrp, dgrp, fgrp;
  int rc, flag, i, nc, ns, nd, nf, nnew, crank, srank, drank, frank;

  int *ranks_cc, *ranks_df, *ranks_ff;

  // Shrink/remove dead process/es
  DLOG(INFO) << "Before Shrinking (mpix_comm_replace)";
  if (comm == MPI_COMM_NULL)
    DLOG(INFO) << "Null Communicator provided (mpix_comm_replace)";

  CPUTimer timer;
  timer.Start();
  MPIX_Comm_shrink(comm, &shrinked);
  LOG(INFO) << "Shrink_time_fault: " << timer.MicroSeconds() << " micro_seconds";
  DLOG(INFO) << "After Shrinking (mpix_comm_replace)";
  MPI_Comm_rank(comm, &crank);
  MPI_Comm_size(comm, &nc);
  MPI_Comm_size(shrinked, &ns);
  if(ns == nc)
    return 0;
  MPI_Comm_group(comm, &cgrp);
  MPI_Comm_group(shrinked, &sgrp);
  MPI_Comm_group(MPI_COMM_WORLD, &fgrp);
  // MPI_Comm_group(first_comm, &fgrp);
  // Diff is the failed ranks

  MPI_Group_difference(fgrp, sgrp, &dgrp); MPI_Group_size(dgrp, &nd);
  ranks_ff = (int*)malloc(nd * sizeof(int));
  ranks_df = (int*)malloc(nd * sizeof(int));
  for (i = 0; i < nd; i++)
    ranks_df [i] = i;

  MPI_Group_translate_ranks(dgrp, nd, ranks_df, fgrp, ranks_ff);
  for(i=0; i< nd; i++) {
    DLOG(INFO)<< "Last Rank Failed: " << ranks_ff[i];
    int last_size = ns + 1;
    int temp_last_rank = -1;
    bool rank_exists = false;
    for(int j = 0; j < size_rank_pair_vec.size(); j++) {
      if(ranks_ff[i] == size_rank_pair_vec[j].second)
        rank_exists = true;
    }
    if(!rank_exists) {
      size_rank_pair_vec.push_back(std::make_pair(last_size, ranks_ff[i]));
    }
  }

  int_pair_vectype::iterator itr;
  for(itr = size_rank_pair_vec.begin(); itr != size_rank_pair_vec.end(); itr++) {
    DLOG(INFO) << "Contents of SzRnkPairVec: Last Size "
              << (*itr).first << " Failed Rank " << (*itr).second;
  }

  // Set Error Handler: instantiated during MPI init
  // MPI_Comm_set_errhandler(shrinked, errh);
  MPI_Comm_size(shrinked, &ns);
  MPI_Comm_rank(shrinked, &srank);

  DLOG (INFO) << "Shrinking Phase: shrunk comm size:" << ns;
  caffe::mpi::new_size = ns;

  // All agree on the reduced size of communicator; // happens in dup_comm.
  timer.Start();
  flag = MPIX_Comm_agree(shrinked, &caffe::mpi::new_size);
  LOG(INFO) << "Shrink_agree_time: " << timer.MicroSeconds() << " micro_seconds";
  if(flag == MPI_SUCCESS) {
    DLOG(INFO) << "All Agree on reduced Comm size, new rank:" << srank <<" , old rank " << crank;
    // swap instead of duplicate
    if(*pnewcomm  == MPI_COMM_NULL)
      DLOG(INFO) << "Empty Communicator for replacement";
    else
      DLOG(INFO) << "Not Empty Communicator for replacement";
    flag = duplicate_comm(pnewcomm, shrinked);
    MPI_Comm_free(&shrinked);
  }
  return flag;
}

MPI_Comm get_working_comm() {
  if(caffe::mpi::wcomm == MPI_COMM_NULL) {
    DLOG(INFO) << "Working Comm is NULL, fixing communicator";
    MPI_Comm temp_comm;
    // TO DO: Fix This
    fix_communicator(&temp_comm);
    caffe::mpi::wcomm = temp_comm;
  }
  return wcomm;
}

// duplicate communicator
int duplicate_comm(MPI_Comm* newcomm, MPI_Comm comm)
{
  int rc;
  int flag;
  int temp_rank, temp_size;
  MPI_Comm* temp_newcomm;
  temp_size = caffe::mpi::comm_size(comm);

  if(comm == MPI_COMM_NULL)
  {
    DLOG(INFO) << "Primary Communicator is Null\n";
  }
  rc = MPI_Comm_dup(comm, newcomm);
  flag = (MPI_SUCCESS == rc);

  MPIX_Comm_agree(comm, &flag);
  if(!flag) {
    caffe::mpi::error_report(rc, &comm);
    if(rc == MPI_SUCCESS) {
      DLOG(INFO) << "Duplication of communicator failed";
      MPI_Comm_free(newcomm);
      rc = MPIX_ERR_PROC_FAILED;
      caffe::mpi::error_report(rc, &comm);
    }
  }

  temp_size = caffe::mpi::comm_size(*newcomm);
  return rc;
}

// Use the same err_code (pointer), rather than copy
void error_report(int err_code, MPI_Comm* Comm)
{
  MPI_Comm comm_ = *Comm;
  char err_string_[MPI_MAX_ERROR_STRING];
  int err_slen_, err_code_;

  err_code_ = err_code;
  MPI_Error_string(err_code_, err_string_, &err_slen_);
  std::cout << "Error Report: " << err_code_ << " ," << err_string_ << std::endl;
}

static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...)
{
  int flag;
  MPI_Comm comm = *pcomm;
  int err_ = *perr;
  char err_string[MPI_MAX_ERROR_STRING];
  int i, rank, size, nf, len, eclass;
  MPI_Group group_c, group_f;
  int *ranks_gc, *ranks_gf;

  MPI_Error_class(err_, &eclass);
  if (MPIX_ERR_PROC_FAILED != eclass) {
    // Not Aborting but continuing
    MPI_Abort(comm, err_);
  }

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPIX_Comm_failure_ack(comm);
  MPIX_Comm_failure_get_acked(comm, &group_f);
  MPI_Group_size(group_f, &nf);
  MPI_Error_string(err_, err_string, &len);
  DLOG(INFO) << "Rank " << rank << "/" << size << ": Notified of error "
    << err_string << "." << nf << "found dead: {";

  ranks_gf = (int*)malloc(nf * sizeof(int));
  ranks_gc = (int*)malloc(nf * sizeof(int));
  MPI_Comm_group(comm, &group_c);
  for(i = 0; i < nf; i++)
    ranks_gf[i] = i;
  MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);
  for(i = 0; i < nf; i++)
    std::cout << ranks_gc[i];
  std::cout << "}\n";
}

void fix_communicator(MPI_Comm* comm)
{
  MPI_Comm rcomm;
  int flag, rsize, wsize, flag2;
  int wb_rank, wa_rank;
  // flag = mpix_comm_replace(wcomm, &rcomm);
  //
  // flag : false; MPIX_Comm_agree (failed on new communicator size);
  flag = mpix_comm_replace(*comm, &rcomm);

  rsize = caffe::mpi::comm_size(rcomm);
  wsize = caffe::mpi::comm_size(*comm);
  wb_rank = caffe::mpi::comm_rank(*comm);
  if (rsize != wsize) {
    DLOG(INFO) << "Working comm size (before switch): " << wsize << ", rank" << wb_rank;
    MPIX_Comm_revoke(*comm);
    MPI_Comm_free(comm);
    flag2 = duplicate_comm(comm, rcomm);
    if (flag != MPI_SUCCESS) {
      LOG(INFO) << "NO agreemeent after communicator fix";
      MPI_Comm_free(&rcomm);
      MPI_Abort(caffe::mpi::wcomm, flag);
    }
    MPI_Comm_set_errhandler(*comm, caffe::mpi::errh);
    caffe::mpi::wcomm = *comm;

    wsize = caffe::mpi::comm_size(*comm);
    wa_rank = caffe::mpi::comm_rank(*comm);
    DLOG(INFO) << "Working comm size (after switch): "
               << wsize << " , rank" << wa_rank;
    MPI_Comm_free(&rcomm);
  }
}

#else /*CAFFE_FT*/

void allreduce_copy(const float& sendbuf, float& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 float)");
  }
}

void allreduce_copy(const double& sendbuf, double& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 double)");
  }
}

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)sendbuf, recvbuf, count,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy float)");
  }
}

void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)sendbuf, recvbuf, count,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy double)");
  }
}

void allreduce(float& buffer, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, &buffer, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 float)");
  }
}

void allreduce(double& buffer, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, &buffer, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 double)");
  }
}

void allreduce(float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, buffer, count,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce float)");
  }
}

void allreduce(double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, buffer, count,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce double)");
  }
}

/*
void iallreduce(MPI_Request &request, float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Iallreduce(MPI_IN_PLACE, buffer, count,
              MPI_FLOAT, op, comm, &request)) {
    throw std::runtime_error("MPI_Iallreduce failed (allreduce float)");
  }
}

void iallreduce(MPI_Request &request, double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Iallreduce(MPI_IN_PLACE, buffer, count,
              MPI_DOUBLE, op, comm, &request)) {
    throw std::runtime_error("MPI_Iallreduce failed (allreduce double)");
  }
}
*/
void bcast(std::vector<int> &buffer, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(&buffer[0], buffer.size(), MPI_INT, root, comm)) {
    throw std::runtime_error("MPI_Bcast vector<int> failed");
  }
}

void bcast(float* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_FLOAT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

bool testall(std::vector<MPI_Request> &requests) {
  int flag = 0;
  if (MPI_SUCCESS != MPI_Testall(requests.size(), &requests[0], &flag, MPI_STATUSES_IGNORE)) {
    throw std::runtime_error("MPI_Waitall failed");
  }
  return flag;
}

void waitall(std::vector<MPI_Request> &requests) {
  if (MPI_SUCCESS != MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE)) {
    throw std::runtime_error("MPI_Waitall failed");
  }
}

bool test(MPI_Request &request) {
  int flag = 0;
  if (MPI_SUCCESS != MPI_Test(&request, &flag, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Test failed");
  }
  return flag;
}

void bcast(double* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}
/*
void send(const float* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Send(buffer, count, MPI_FLOAT, dest, tag, comm)) {
    throw std::runtime_error("MPI_Send failed (float)");
  }
}

void send(const double* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Send(buffer, count, MPI_DOUBLE, dest, tag, comm)) {
    throw std::runtime_error("MPI_Send failed (double)");
  }
}

void recv(float *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Recv(buffer, count, MPI_FLOAT, source, tag, comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Recv failed (float)");
  }
}

void recv(double *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Recv(buffer, count, MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Recv failed (double)");
  }
}

void sendrecv(const int *sendbuf, int sendcount, int dest, int sendtag,
    int *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm)
{
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Sendrecv(sendbuf, sendcount, MPI_INT, dest, sendtag,
        recvbuf, recvcount, MPI_INT, source, recvtag,
        comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Sendrecv failed (int)");
  }
}

void sendrecv(const signed char *sendbuf, int sendcount, int dest, int sendtag,
    signed char *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm)
{
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Sendrecv(sendbuf, sendcount, MPI_CHAR, dest, sendtag,
        recvbuf, recvcount, MPI_CHAR, source, recvtag,
        comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Sendrecv failed (char)");
  }
}

void sendrecv(const float *sendbuf, int sendcount, int dest, int sendtag,
    float *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm)
{
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Sendrecv(sendbuf, sendcount, MPI_FLOAT, dest, sendtag,
        recvbuf, recvcount, MPI_FLOAT, source, recvtag,
        comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Sendrecv failed (float)");
  }
}

void sendrecv(const double *sendbuf, int sendcount, int dest, int sendtag,
    double *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm)
{
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Sendrecv(sendbuf, sendcount, MPI_DOUBLE, dest, sendtag,
        recvbuf, recvcount, MPI_DOUBLE, source, recvtag,
        comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Sendrecv failed (double)");
  }
}

void isend(MPI_Request &request, const signed char* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Isend(buffer, count, MPI_CHAR, dest, tag, comm, &request)) {
    throw std::runtime_error("MPI_Isend failed (signed char)");
  }
}

void isend(MPI_Request &request, const int* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Isend(buffer, count, MPI_INT, dest, tag, comm, &request)) {
    throw std::runtime_error("MPI_Isend failed (int)");
  }
}

void isend(MPI_Request &request, const float* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Isend(buffer, count, MPI_FLOAT, dest, tag, comm, &request)) {
    throw std::runtime_error("MPI_Isend failed (float)");
  }
}

void isend(MPI_Request &request, const double* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Isend(buffer, count, MPI_DOUBLE, dest, tag, comm, &request)) {
    throw std::runtime_error("MPI_Isend failed (double)");
  }
}

void irecv(MPI_Request &request, signed char *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Irecv(buffer, count, MPI_CHAR, source, tag, comm, &request)) {
    throw std::runtime_error("MPI_Irecv failed (signed char)");
  }
}

void irecv(MPI_Request &request, int *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Irecv(buffer, count, MPI_INT, source, tag, comm, &request)) {
    throw std::runtime_error("MPI_Irecv failed (int)");
  }
}

void irecv(MPI_Request &request, float *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Irecv(buffer, count, MPI_FLOAT, source, tag, comm, &request)) {
    throw std::runtime_error("MPI_Irecv failed (float)");
  }
}

void irecv(MPI_Request &request, double *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Irecv(buffer, count, MPI_DOUBLE, source, tag, comm, &request)) {
    throw std::runtime_error("MPI_Irecv failed (double)");
  }
}
*/
#endif /*CAFFE_FT update this*/

#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe
