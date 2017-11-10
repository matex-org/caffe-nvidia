#ifndef CAFFE_MPI_HPP_
#define CAFFE_MPI_HPP_

#ifdef USE_MPI
#include <mpi.h>
// #ifdef CAFFE_FT
#include <glog/logging.h>
#include <mpi-ext.h>
#include <signal.h>
// #endif /*CAFFE_FT*/
#endif

#ifdef CAFFE_FT
#include <tuple>
#include <algorithm>
#include <utility>
#endif /*CAFFE_FT*/
#include <string>
#include <vector>

#define NO_MPI LOG(FATAL) << "Cannot use MPI unless USE_MPI is enabled during make."

namespace caffe {
namespace mpi {

#ifdef USE_MPI

#ifdef CAFFE_FT
extern MPI_Comm wcomm, rcomm, first_comm;
// extern int fault_global_flag;
// extern int* last_ranks_failed;
extern int old_size;
extern int new_size;
extern int last_rank_failed;
extern bool solver_completed;
// Last failed size, last failed (original) rank.

typedef std::pair<int, int> int_pairtype;
typedef std::vector<int_pairtype> int_pair_vectype;
extern int_pair_vectype size_rank_pair_vec;

#endif

extern MPI_Comm default_comm_;
MPI_Comm get_comm_default();
void set_comm_default(MPI_Comm comm=MPI_COMM_WORLD);

void init(int *argc, char ***argv);
bool initialized();
int query_thread();
void finalize();

MPI_Comm comm_dup(MPI_Comm comm=MPI_COMM_NULL);
MPI_Comm comm_split(int color, int key, MPI_Comm comm=MPI_COMM_NULL);
MPI_Comm comm_create(MPI_Group group, MPI_Comm comm=MPI_COMM_NULL);
MPI_Comm comm_create(const std::vector<int> &incl, MPI_Comm comm=MPI_COMM_NULL);
void comm_free(MPI_Comm comm);
int comm_rank(MPI_Comm comm=MPI_COMM_NULL);
int comm_size(MPI_Comm comm=MPI_COMM_NULL);
int node_rank(MPI_Comm comm=MPI_COMM_NULL);
int node_size(MPI_Comm comm=MPI_COMM_NULL);

template <typename Dtype>
MPI_Datatype datatype();

#ifdef CAFFE_FT
struct FTCommunicator {
MPI_Errhandler errh; // Error Handler;
MPI_Comm working_comm, split_comm;
//static void verbose_errhandler(MPI_Comm* comm, int* err, ...);
};

// for global working comm and recovery comm.
extern MPI_Comm wcomm, rcomm;
// error message
extern char err_str[MPI_MAX_ERROR_STRING];
extern int err_strlen;

void completed(bool comp);
void update_faulted_processes(int faulted_rank);

int mpix_comm_replace(MPI_Comm comm, MPI_Comm* newcomm);
MPI_Comm get_working_comm();
int duplicate_comm(MPI_Comm* newcomm, MPI_Comm comm=MPI_COMM_NULL);
void error_report(int err_code, MPI_Comm* comm);
static void verbose_errhandler(MPI_Comm* comm, int* err, ...);
void fix_communicator(MPI_Comm* comm);

void allreduce_copy(const float& sendbuf, float& recvbuf,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce_copy(const double& sendbuf, double& recvbuf,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

// Note: Return Type: std::tuple<int, bool> (return_val_fromMPICALL, comm_repaired)

// int allreduce(float& recvbuf, MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
// int allreduce(double& recvbuf, MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
std::tuple<int,bool> allreduce(float& recvbuf, MPI_Op op=MPI_SUM
        , MPI_Comm comm=MPI_COMM_NULL);
std::tuple<int, bool> allreduce(double& recvbuf, MPI_Op op=MPI_SUM
        , MPI_Comm comm=MPI_COMM_NULL);

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

// int allreduce(float* buffer, int count,
//        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
// int allreduce(double* buffer, int count,
//        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
std::tuple<int, bool> allreduce(float* buffer, int count
        , MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
std::tuple<int, bool> allreduce(double* buffer, int count
        , MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void bcast(int* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(float* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(double* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);
#else
void allreduce_copy(const float& sendbuf, float& recvbuf,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce_copy(const double& sendbuf, double& recvbuf,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void allreduce(float& recvbuf, MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce(double& recvbuf, MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void allreduce(float* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void allreduce(double* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

void iallreduce(MPI_Request &request, float* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);
void iallreduce(MPI_Request &request, double* buffer, int count,
        MPI_Op op=MPI_SUM, MPI_Comm comm=MPI_COMM_NULL);

bool testall(std::vector<MPI_Request> &requests);
void waitall(std::vector<MPI_Request> &requests);
bool test(MPI_Request &request);

void bcast(std::vector<int> &buffer, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(float* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);
void bcast(double* buffer, int count, int root=0, MPI_Comm comm=MPI_COMM_NULL);

void send(const float *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void send(const double *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

void recv(float *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void recv(double *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

void sendrecv(const int *sendbuf, int sendcount, int dest, int sendtag,
    int *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);
void sendrecv(const signed char *sendbuf, int sendcount, int dest, int sendtag,
    signed char *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);
void sendrecv(const float *sendbuf, int sendcount, int dest, int sendtag,
    float *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);
void sendrecv(const double *sendbuf, int sendcount, int dest, int sendtag,
    double *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm=MPI_COMM_NULL);

void isend(MPI_Request &request, const signed char *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void isend(MPI_Request &request, const int *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void isend(MPI_Request &request, const float *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void isend(MPI_Request &request, const double *buf, int count, int dest=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

void irecv(MPI_Request &request, signed char *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void irecv(MPI_Request &request, int *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void irecv(MPI_Request &request, float *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);
void irecv(MPI_Request &request, double *buf, int count, int source=0, int tag=1234, MPI_Comm comm=MPI_COMM_NULL);

#endif /* CAFFE_FT */

#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe

#endif // CAFFE_MPI_HPP_
