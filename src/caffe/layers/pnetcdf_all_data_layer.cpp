#include <stdint.h>
#include <string.h>

#include <vector>

#if USE_PNETCDF
#include <pnetcdf.h>
#endif

#ifdef CAFFE_FT
#include <memory>
#include <algorithm>
#endif

#define STRIDED 0

#include <boost/thread.hpp>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/pnetcdf_all_data_layer.hpp"
#include "caffe/mpi.hpp"
#include <assert.h>
//#if CAFFE_FT
//#include <mpi-ext.h>
//#endif
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
PnetCDFAllDataLayer<Dtype>::PnetCDFAllDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    current_row_(0),
    max_row_(0),
    datum_shape_(),
    data_(),
    label_(),
    data_char_count_(0),
    label_int_count_(0),
#ifdef CAFFE_FT
    padd_max_row_(0),
    padd_data_(),
    padd_label_(),
    padd_data_char_count_(0),
    padd_label_int_count_(0),
#endif
    row_mutex_(),
    comm_(),
    comm_rank_(),
    comm_size_() {
#ifdef CAFFE_FT
  comm_ = caffe::mpi::get_working_comm();
  // MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);
  std::cout << "Working Comm PNETCDFALLDATALAYER.\n";
#else
  comm_ = caffe::mpi::comm_dup();
#endif
  comm_rank_ = caffe::mpi::comm_rank(comm_);
  comm_size_ = caffe::mpi::comm_size(comm_);
  LOG(INFO) << "Rank PnetCDF file: " << comm_rank_
      << " Size: " << comm_size_;

}

template <typename Dtype>
PnetCDFAllDataLayer<Dtype>::~PnetCDFAllDataLayer() {
  this->StopInternalThread();
}

static void errcheck(int retval) {
#if USE_PNETCDF
  if (NC_NOERR != retval) {
    LOG(FATAL) << "pnetcdf error: " << ncmpi_strerror(retval);
  }
#else
  NO_PNETCDF;
#endif
}

template <typename Dtype>
inline static Dtype prod(vector<Dtype> vec) {
  Dtype val = 1;
  for (size_t i=0; i<vec.size(); i++) {
    val *= vec[i];
  }
  return val;
}

#ifdef CAFFE_FT
template <typename Dtype>
std::tuple<int, bool> PnetCDFAllDataLayer<Dtype>::fix_comm_error(MPI_Comm* comm, float* x)
{
  // MPI_Comm* tcomm = comm;
  std::tuple<int, bool> retVal;
  int trank, tsize;
  caffe::mpi::fix_communicator(comm);
  // *comm = caffe::mpi::get_working_comm();
  this->comm_ = caffe::mpi::get_working_comm();
  // trank = caffe::mpi::comm_rank(*tcomm);
  this->comm_rank_ = caffe::mpi::comm_rank(this->comm_);
  // tsize = caffe::mpi::comm_size(*tcomm);
  this->comm_size_ = caffe::mpi::comm_size(this->comm_);
  DLOG(INFO) << "Communicator Fixed: Rank: " << this->comm_rank_
      << ", Size: " << this->comm_size_;

  std::get<1>(retVal) = true;
  // std::get<0>(retVal) =  MPI_Allreduce(MPI_IN_PLACE, x, 1, MPI_FLOAT, MPI_SUM, *comm);
  std::get<0>(retVal) =  MPI_Allreduce(MPI_IN_PLACE, x, 1, MPI_FLOAT, MPI_SUM, this->comm_);
  if(std::get<0>(retVal) != MPI_SUCCESS) {
    DLOG(INFO) << "ERROR OCCURED BEFORE FILE ACCESS. Could not recover. Aborting";
    caffe::mpi::error_report(std::get<0>(retVal), &(this->comm_));
    // MPI_Abort(*comm, std::get<0>(retVal));
    MPI_Abort(this->comm_, std::get<0>(retVal));
  }
  // comm_ = caffe::mpi::get_working_comm();
  // comm_rank_ = caffe::mpi::comm_rank(*comm);
  // comm_size_ = caffe::mpi::comm_size(*comm);
  return retVal;
}
#endif

template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::load_pnetcdf_file_data(const string& filename) {
#if USE_PNETCDF
#if STRIDED
  LOG(INFO) << "Loading PnetCDF file, strided: " << filename;
#else
  LOG(INFO) << "Loading PnetCDF file: " << filename;
#endif

  int rank = comm_rank_;
  int size = comm_size_;
  int retval;
  int ncid;
  int ndims;
  int nvars;
  int ngatts;
  int unlimdim;
  MPI_Offset total;
  MPI_Offset count_;
  MPI_Offset remain;
  MPI_Offset start;
  MPI_Offset stop;

  #ifdef CAFFE_FT
  // int rc, rc2;
  std::tuple<int, bool> retVal1, retVal2;

  float xallreduce = 1.0;
// check if communicator is valid

  std::get<1>(retVal1) = false;
  std::get<0>(retVal1) = MPI_Allreduce(MPI_IN_PLACE, &xallreduce, 1, MPI_FLOAT,
              MPI_SUM, this->comm_);
  if(std::get<0>(retVal1) != MPI_SUCCESS)
  {
    retVal2 = fix_comm_error(&comm_, &xallreduce);
    DLOG(INFO) << "ERROR OCCURED BEFORE FILE ACCESS";
  }
  DLOG(INFO) << "Test AllReduce Value: " << xallreduce;
  retval = ncmpi_open(this->comm_, filename.c_str(),
          NC_NOWRITE, MPI_INFO_NULL, &ncid);
  #else
  retval = ncmpi_open(this->comm_, filename.c_str(),
          NC_NOWRITE, MPI_INFO_NULL, &ncid);
  #endif
  errcheck(retval);
  rank = this->comm_rank_;
  size = this->comm_size_;

  DLOG(INFO) << "PnetCDF After Opening File:-------w rank: "
             << this->comm_rank_ << ", size:" << this->comm_size_;

  retval = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &unlimdim);
  errcheck(retval);

  retval = ncmpi_inq_dimlen(ncid, unlimdim, &total);
  errcheck(retval);

  count_ = total / size;
  remain = total % size;
#if STRIDED
  start = rank;
  stop = rank; // dummy value, not used
  if (rank < remain) {
      count_ += 1;
  }
#else
  start = rank * count_;
  stop = rank * count_ + count_;
  if (rank < remain) {
    start += rank;
    stop += rank + 1;
  } else {
    start += remain;
    stop += remain;
  }
#endif

  DLOG(INFO) << "ncid " << ncid;
  DLOG(INFO) << "ndims " << ndims;
  DLOG(INFO) << "nvars " << nvars;
  DLOG(INFO) << "ngatts " << ngatts;
  DLOG(INFO) << "unlimdim " << unlimdim;
  DLOG(INFO) << "total images " << total;
  DLOG(INFO) << "start " << start;
  DLOG(INFO) << "stop " << stop;

  for (int varid = 0; varid < nvars; varid++) {
    int vartype;
    int varndims;
    vector<int> vardimids;
    vector<MPI_Offset> count;
    vector<MPI_Offset> offset;
    vector<MPI_Offset> stride;
    MPI_Offset chunksize = 2147483647L;
    MPI_Offset prodcount;

    retval = ncmpi_inq_vartype(ncid, varid, &vartype);
    errcheck(retval);

    retval = ncmpi_inq_varndims(ncid, varid, &varndims);
    errcheck(retval);

    vardimids.resize(varndims);
    count.resize(varndims);
    offset.resize(varndims);
    stride.resize(varndims);

    retval = ncmpi_inq_vardimid(ncid, varid, &vardimids[0]);
    errcheck(retval);

    for (int i = 0; i < varndims; i++) {
      retval = ncmpi_inq_dimlen(ncid, vardimids[i], &count[i]);
      errcheck(retval);
      offset[i] = 0;
      stride[i] = 1;
      if (count[i] > chunksize) {
        LOG(FATAL) << "dimension is too large for Blob";
      }
    }
    // MPI-IO can only read 2GB chunks due to "int" interface for indices
#if STRIDED
    count[0] = count_;
    offset[0] = start;
    stride[0] = size;
#else
    count[0] = stop-start;
    offset[0] = start;
#endif
    prodcount = prod(count);

    if (NC_BYTE == vartype) {
      // this->data_ = shared_ptr<signed char>(new signed char[prodcount]);
      //this->data_.reserve(prodcount);
      this->data_.resize(prodcount);
      datum_shape_.resize(4);
      datum_shape_[0] = 1;
      datum_shape_[1] = count[1];
      datum_shape_[2] = count[2];
      datum_shape_[3] = count[3];
      DLOG(INFO) << "datum_shape_[0] " << datum_shape_[0];
      DLOG(INFO) << "datum_shape_[1] " << datum_shape_[1];
      DLOG(INFO) << "datum_shape_[2] " << datum_shape_[2];
      DLOG(INFO) << "datum_shape_[3] " << datum_shape_[3];
      if (prodcount < chunksize) {
        LOG(INFO) << "reading PnetCDF data whole " << count[0];
        LOG(INFO) << "offset={"<<offset[0]<<","<<offset[1]<<","<<offset[2]<<","<<offset[3]<<"}";
        LOG(INFO) << "count={"<<count[0]<<","<<count[1]<<","<<count[2]<<","<<count[3]<<"}";
#if STRIDED
        //retval = ncmpi_get_vars_schar_all(ncid, varid, &offset[0],
        //    &count[0], &stride[0], this->data_.get());
        retval = ncmpi_get_vars_schar_all(ncid, varid, &offset[0],
			&count[0], &stride[0], &(this->data_[0]));
            // &count[0], &stride[0], this->data_.get());
#else
        // retval = ncmpi_get_vara_schar_all(ncid, varid, &offset[0],
        //     &count[0], this->data_.get());
        retval = ncmpi_get_vara_schar_all(ncid, varid, &offset[0],
			&count[0], &(this->data_[0]));
            // &count[0], this->data_.get());
#endif
        data_char_count_ += prodcount;
        errcheck(retval);
      }
      else {
        vector<MPI_Offset> newoffset = offset;
        vector<MPI_Offset> newcount = count;
        MPI_Offset data_offset = 0;
        newcount[0] = 1;
        MPI_Offset newprodcount = prod(newcount);
        newcount[0] = chunksize/newprodcount;
        newprodcount = prod(newcount);
        if (newprodcount >= chunksize) {
          LOG(FATAL) << "newprodcount >= chunksize";
        }
        MPI_Offset cur = 0;
        //shared_ptr<signed char> chunk = shared_ptr<signed char>(
        //    new signed char[newprodcount]);
        vector<signed char> chunk(newprodcount);

        // = shared_ptr<signed char>(
        //    new signed char[newprodcount]);

        while (cur < count[0]) {
          if (cur+newcount[0] > count[0]) {
            newcount[0] = count[0]-cur;
            newprodcount = prod(newcount);
          }
          LOG(INFO) << "reading data chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
          //retval = ncmpi_get_vars_schar_all(ncid, varid, &newoffset[0],
          //    &newcount[0], &stride[0], chunk.get());
          retval = ncmpi_get_vars_schar_all(ncid, varid, &newoffset[0],
		  	  &newcount[0], &stride[0], &chunk[0];
              // &newcount[0], &stride[0], chunk.get());
#else
          // retval = ncmpi_get_vara_schar_all(ncid, varid, &newoffset[0],
          //    &newcount[0], chunk.get());
          retval = ncmpi_get_vara_schar_all(ncid, varid, &newoffset[0],
		      &newcount[0], &chunk[0]);
              // &newcount[0], chunk.get());
#endif
          errcheck(retval);
          // memcpy(this->data_.get() + data_offset, chunk.get(), newprodcount);
          memcpy(&(this->data_[0]) + data_offset, &chunk[0], newprodcount);

          data_char_count_ += newprodcount;
          cur += newcount[0];
#if STRIDED
          newoffset[0] += newcount[0]*size;
#else
          newoffset[0] += newcount[0];
#endif
          data_offset += newprodcount;
        }
      }
    }
    else if (NC_INT == vartype && this->output_labels_) {
      max_row_ = count[0];
      // this->label_ = shared_ptr<int[]>(new int[max_row_]);
      // this->label_.reserve(max_row_);
      this->label_.resize(max_row_);
      LOG(INFO) << "PnetCDF max_row_ = " << max_row_;
      if (prodcount < chunksize) {
        LOG(INFO) << "reading PnetCDF label whole " << count[0];
#if STRIDED
        //retval = ncmpi_get_vars_int_all(ncid, varid, &offset[0],
        //    &count[0], &stride[0], this->label_.get());
        retval = ncmpi_get_vars_int_all(ncid, varid, &offset[0],
            // &count[0], &stride[0], this->label_.get());
			&count[0], &stride[0], &(this->label_[0]));
#else
        // retval = ncmpi_get_vara_int_all(ncid, varid, &offset[0],
        //     &count[0], this->label_.get());
        retval = ncmpi_get_vara_int_all(ncid, varid, &offset[0],
            // &count[0], this->label_.get());
			&count[0], &(this->label_[0]));
#endif
        label_int_count_ += count[0]; //
        errcheck(retval);
      }
      else {
        vector<MPI_Offset> newoffset = offset;
        vector<MPI_Offset> newcount = count;
        MPI_Offset data_offset = 0;
        newcount[0] = 1;
        MPI_Offset newprodcount = prod(newcount);
        newcount[0] = chunksize/newprodcount;
        newprodcount = prod(newcount);
        if (newprodcount >= chunksize) {
          LOG(FATAL) << "newprodcount >= chunksize";
        }
        MPI_Offset cur = 0;
        // shared_ptr<int> chunk = shared_ptr<int>(new int[newprodcount]);
        vector<int> chunk(newprodcount);

        while (cur < count[0]) {
          if (cur+newcount[0] > count[0]) {
            newcount[0] = count[0]-cur;
            newprodcount = prod(newcount);
          }
          LOG(INFO) << "reading label chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
          retval = ncmpi_get_vars_int_all(ncid, varid, &newoffset[0],
              // &newcount[0], &stride[0], chunk.get());
			  &newcount[0], &stride[0], &chunk[0]);
		  // retval = ncmpi_get_vars_int_all(ncid, varid, &newoffset[0],
          //     &newcount[0], &stride[0], chunk.get());
#else
          //retval = ncmpi_get_vara_int_all(ncid, varid, &newoffset[0],
          //    &newcount[0], chunk.get());
          retval = ncmpi_get_vara_int_all(ncid, varid, &newoffset[0],
              // &newcount[0], chunk.get());
			  &newcount[0], &chunk[0]);
#endif
          errcheck(retval);
          //memcpy(this->label_.get() + data_offset, chunk.get(), newprodcount);
          memcpy(&(this->label_[0]) + data_offset, &chunk[0], newprodcount);

          label_int_count_ += newcount[0];
          cur += newcount[0];
#if STRIDED
          newoffset[0] += newcount[0]*size;;
#else
          newoffset[0] += newcount[0];
#endif
          data_offset += newprodcount;
        }
      }
    }
    else {
      LOG(FATAL) << "unknown data type";
    }
  }

  retval = ncmpi_close(ncid);
  errcheck(retval);

  DLOG(INFO) << "Total Data Char Count:" << data_char_count_;
  DLOG(INFO) << "Total Label Int Count:" << label_int_count_;

  {
    const int batch_size = this->layer_param_.data_param().batch_size();
    Dtype label_sum = 0;
    for (int i=0; i<batch_size; i++) {
      // label_sum += *(this->label_.get()+i);
      label_sum += this->label_[i];
    }
    // caffe::mpi::allreduce(label_sum);
    LOG(INFO) << "Label Sum (Before AllReduce): " << label_sum;
    caffe::mpi::allreduce(&label_sum, 1, MPI_SUM, this->comm_ );
    LOG(INFO) << "Label Sum (After AllReduce): "
              << label_sum/this->comm_size_;
  }
#else
  NO_PNETCDF;
#endif
}

////////////////////////////////////////
#ifdef CAFFE_FT
template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::reload_pnetcdf_file_data(const string& filename) {
#if USE_PNETCDF
  #if STRIDED
    LOG(INFO) << "Loading PnetCDF file, strided: " << filename;
  #else
    LOG(INFO) << "Loading PnetCDF file: " << filename;
  #endif

  int retval;
  int ncid;
  int ndims;
  int nvars;
  int ngatts;
  int unlimdim;
  MPI_Offset total;
  MPI_Offset count_;
  MPI_Offset remain;
  MPI_Offset faulted_start;
  MPI_Offset faulted_stop;

  std::tuple<int, bool> retVal1, retVal2;
  float xallreduce = 1.0;
  // check if communicator is valid
  comm_ = caffe::mpi::get_working_comm();
  std::get<1>(retVal1) = false;
  std::get<0>(retVal1) = MPI_Allreduce(MPI_IN_PLACE, &xallreduce, 1, MPI_FLOAT,
              MPI_SUM, comm_);
  if(std::get<0>(retVal1) != MPI_SUCCESS)
  {
    retVal2 = fix_comm_error(&comm_, &xallreduce);
    DLOG(INFO) << "ERROR OCCURED BEFORE FILE ACCESS";
  }
  DLOG(INFO) << "Test AllReduce Value: " << xallreduce;
  retval = ncmpi_open(comm_, filename.c_str(),
          NC_NOWRITE, MPI_INFO_NULL, &ncid);
  errcheck(retval);

  DLOG(INFO) << "PnetCDF After Opening File:-------w rank: "
             << this->comm_rank_ << ", size:" << this->comm_size_;

  retval = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &unlimdim);
  errcheck(retval);

  retval = ncmpi_inq_dimlen(ncid, unlimdim, &total);
  errcheck(retval);

  // Make universal for multiple faults
  // if(caffe::mpi::last_rank_failed >= 0) {
  // Size_Rank_Pair_Vec (BeforeSize_Rank Failed)
  auto srp_vec = std::make_shared<caffe::mpi::int_pair_vectype>(
                  caffe::mpi::size_rank_pair_vec);

  if(caffe::mpi::size_rank_pair_vec.size() >= 0) {
    // [first:start, second:stop]
    typedef std::pair<MPI_Offset, MPI_Offset> offset_pair_type;
#if STRIDED
    std::vector<offset_pair_type> start_count_vec;
#else
    vector<offset_pair_type> start_stop_vec;
#endif
    // Non Strided version only
    MPI_Offset f_start, f_stop, f_count, f_remain
             , p_start, p_stop, p_count, p_remain
             , temp_start, temp_stop, temp_count, temp_remain;

    std::size_t frank_sz = srp_vec->size();
    for(int i = 0; i < frank_sz; i++) {
      f_count = total/(*srp_vec)[i].first;
      DLOG(INFO) << "___________f_count : " << f_count;
      f_remain = total % (*srp_vec)[i].first;
      DLOG(INFO) << "___________f_remain : " << f_remain;
      DLOG(INFO) << "___________srp_size : " << (*srp_vec)[i].first;

#if STRIDED
      f_start = (*srp_vec)[i].second;
      // f_stop = (*srp_vec)[i].second;
      if((*srp_vec)[i].second < f_remain)
        f_count +=1;
#else
      f_start = f_count * (*srp_vec)[i].second;
      DLOG(INFO) << "___________f_startI : " << f_start;
      f_stop = f_count * (*srp_vec)[i].second + f_count;
      DLOG(INFO) << "___________f_stopI : " << f_stop;
      this->add_remaining((*srp_vec)[i].second, f_remain, &f_start, &f_stop);

      DLOG(INFO) << "___________f_startII : " << f_start;
      DLOG(INFO) << "___________f_stopII : " << f_stop;
      DLOG(INFO) << "___________srp_rank : " << (*srp_vec)[i].second;
#endif
      for(int j = i+1 ; j < frank_sz; j++) {
          temp_count = (f_stop - f_start)/(*srp_vec)[j].first;
          temp_remain = (f_stop - f_start)%(*srp_vec)[j].first;
          DLOG(INFO) << "f_start ____: " << f_start << " f_stop____: "<< f_stop;

          DLOG(INFO) << "Last Size: " << (*srp_vec)[j-1].first
                     << " Next Size: " << (*srp_vec)[j].first;
          DLOG(INFO) << "Last Rank: " << (*srp_vec)[j-1].second
                     << " ____ Next Rank: " << (*srp_vec)[j].second;
          DLOG(INFO) << "Temp Count____: " << temp_count
                     << " Temp remain____: " << temp_remain;
#if STRIDED
          f_start = f_start + (*srp_vec)[j].second;
          // f_stop = (*srp_vec)[j].second;
          if(*srp_vec[j].second < temp_remain)
            temp_count += 1;
#else
          f_start  = f_start + temp_count * (*srp_vec)[j].second;
          f_stop = f_start + temp_count;
          this->add_remaining((*srp_vec)[j].second, temp_remain, &f_start, &f_stop);
#endif
          DLOG(INFO) << "next f_start ____: " << f_start
                     << " next f_stop____: " << f_stop
                     << " this->comm_size_: " << this->comm_size_;
      }
      // ##
      p_count = (f_stop - f_start) / this->comm_size_;
      p_remain = (f_stop -f_start) % this->comm_size_;
      // p_count = f_stop - f_start;
#if STRIDED
      p_start = f_start + comm_rank_;
      // p_stop = f_start + comm_rank_;
      if (comm_rank_ < p_remain)
        p_count += 1;
#else
      //##
      p_start = f_start + this->comm_rank_ * p_count;
      p_stop = f_start + this->comm_rank_ * p_count + p_count;
      this->add_remaining(this->comm_rank_, p_remain, &p_start, &p_stop);
      // p_start = f_start;
      // p_stop = f_start + p_count;
#endif
      DLOG(INFO) << "last p_start ____: " << p_start
                     << " last p_stop____: " << p_stop;
#if STRIDED
      start_count_vec.push_back(std::make_pair(p_start, p_count));
#else
      start_stop_vec.push_back(std::make_pair(p_start, p_stop));
#endif
      DLOG(INFO) << "ncid " << ncid;
      DLOG(INFO) << "ndims " << ndims;
      DLOG(INFO) << "nvars " << nvars;
      DLOG(INFO) << "ngatts " << ngatts;
      DLOG(INFO) << "unlimdim " << unlimdim;
      DLOG(INFO) << "total images " << total;
      DLOG(INFO) << "padd start (failed)" << p_start;
      DLOG(INFO) << "padd stop (failed)" << p_stop;
      DLOG(INFO) << "------------------------" ;
    }

    // Go through the offset, count pair for each failed rank/s
    // [data_ptr, count]
    typedef std::pair<vector<signed char>
                      , MPI_Offset> data_count_pair_type;
    //typedef std::pair<std::unique_ptr<signed char>
    //                  , MPI_Offset> data_count_pair_type;
    vector<data_count_pair_type> data_chunks;

    // [label_ptr, count]
    //typedef std::pair<shared_ptr<int>
    //                  , MPI_Offset> label_count_pair_type;
    typedef std::pair<vector<int>
                      , MPI_Offset> label_count_pair_type;

    //typedef std::pair<std::unique_ptr<int>
    //                  , MPI_Offset> label_count_pair_type;
    vector<label_count_pair_type> label_chunks;

#if STRIDED
    for(int i = 0; i < start_count_vec.size() ; i++) {
#else
    for(int i = 0; i < start_stop_vec.size() ; i++) {
#endif
      for (int varid = 0; varid < nvars; varid++) {
        int vartype;
        int varndims;
        vector<int> vardimids;
        vector<MPI_Offset> count;
        vector<MPI_Offset> offset;
        vector<MPI_Offset> stride;
        MPI_Offset chunksize = 2147483647L;
        MPI_Offset prodcount;

        retval = ncmpi_inq_vartype(ncid, varid, &vartype);
        errcheck(retval);

        retval = ncmpi_inq_varndims(ncid, varid, &varndims);
        errcheck(retval);

        vardimids.resize(varndims);
        count.resize(varndims);
        offset.resize(varndims);
        stride.resize(varndims);

        retval = ncmpi_inq_vardimid(ncid, varid, &vardimids[0]);
        errcheck(retval);

        vardimids.resize(varndims);
        count.resize(varndims);
        offset.resize(varndims);
        stride.resize(varndims);

        retval = ncmpi_inq_vardimid(ncid, varid, &vardimids[0]);
        errcheck(retval);

        for (int i = 0; i < varndims; i++) {
          retval = ncmpi_inq_dimlen(ncid, vardimids[i], &count[i]);
          errcheck(retval);
          offset[i] = 0;
          stride[i] = 1;
          if (count[i] > chunksize) {
            LOG(FATAL) << "dimension is too large for Blob";
          }
        }
        // MPI-IO can only read 2GB chunks due to "int" interface for indices

#if STRIDED
          // Need to revisit
          // count[0] = padd_count_;
          // offset[0] = padd_start;
          count[0] = start_count_vec[i].second;
          offset[0] = start_count_vec[i].first;
          stride[0] = comm_size_;  // TODO: Need verification this works
#else
          // count[0] = padd_stop - padd_start;
          // offset[0] = padd_start;
          count[0] = start_stop_vec[i].second - start_stop_vec[i].first;
          offset[0] = start_stop_vec[i].first;
#endif
          prodcount = prod(count);

          if (NC_BYTE == vartype) {
            // this->padd_data_ = shared_ptr<signed char>(new signed char[prodcount]
            //                                  , array_deleter<signed char>());
            vector<signed char> temp_padd_data_(prodcount);
            // this->padd_data_.reserve(prodcount);

            datum_shape_.resize(4);
            datum_shape_[0] = 1;
            datum_shape_[1] = count[1];
            datum_shape_[2] = count[2];
            datum_shape_[3] = count[3];
            DLOG(INFO) << "datum_shape_[0] " << datum_shape_[0];
            DLOG(INFO) << "datum_shape_[1] " << datum_shape_[1];
            DLOG(INFO) << "datum_shape_[2] " << datum_shape_[2];
            DLOG(INFO) << "datum_shape_[3] " << datum_shape_[3];
            if (prodcount < chunksize) {
              LOG(INFO) << "reading PnetCDF data whole " << count[0];
              LOG(INFO) << "offset={"<<offset[0]<<","<<offset[1]<<","<<offset[2]<<","<<offset[3]<<"}";
              LOG(INFO) << "count={"<<count[0]<<","<<count[1]<<","<<count[2]<<","<<count[3]<<"}";
#if STRIDED
              //retval = ncmpi_get_vars_schar_all(ncid, varid, &offset[0],
              //    &count[0], &stride[0], this->padd_data_.get());
              retval = ncmpi_get_vars_schar_all(ncid, varid, &offset[0],
                  &count[0], &stride[0], &(temp_padd_data_[0]));
#else
              // retval = ncmpi_get_vara_schar_all(ncid, varid, &offset[0],
              //     &count[0], this->padd_data_.get());
              retval = ncmpi_get_vara_schar_all(ncid, varid, &offset[0],
                  &count[0], &(temp_padd_data_[0]));

#endif
              //this->padd_data_char_count_ += prodcount;
              DLOG(INFO) << "Prodcount Val: " << prodcount << " ,rank:" << comm_rank_;
              data_chunks.push_back(std::make_pair(
                                        temp_padd_data_, prodcount));
              errcheck(retval);
              DLOG(INFO) << "Index Error here?... ";
            }
            else {
              vector<MPI_Offset> newoffset = offset;
              vector<MPI_Offset> newcount = count;
              MPI_Offset data_offset = 0;
              newcount[0] = 1;
              MPI_Offset newprodcount = prod(newcount);
              newcount[0] = chunksize/newprodcount;
              newprodcount = prod(newcount);
              if (newprodcount >= chunksize) {
                LOG(FATAL) << "newprodcount >= chunksize";
              }
              MPI_Offset cur = 0;
              MPI_Offset temp_prodcount = 0;
              //shared_ptr<signed char> chunk = shared_ptr<signed char>(
              //   new signed char[newprodcount], array_deleter<signed char>());
              vector<signed char> chunk(newprodcount);

              while (cur < count[0]) {
                if (cur+newcount[0] > count[0]) {
                  newcount[0] = count[0]-cur;
                  newprodcount = prod(newcount);
                }
                LOG(INFO) << "reading data chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
                //retval = ncmpi_get_vars_schar_all(ncid, varid, &newoffset[0],
                //    &newcount[0], &stride[0], chunk.get());
                retval = ncmpi_get_vars_schar_all(ncid, varid, &newoffset[0],
                    &newcount[0], &stride[0], &chunk[0]);
#else
                // retval = ncmpi_get_vara_schar_all(ncid, varid, &newoffset[0],
                //     &newcount[0], chunk.get());
                retval = ncmpi_get_vara_schar_all(ncid, varid, &newoffset[0],
                    &newcount[0], &chunk[0]);
#endif
                errcheck(retval);
                memcpy(&(temp_padd_data_[0]) + data_offset, &chunk[0], newprodcount);
                DLOG(INFO) << "NewProdCount Val: " << newprodcount;
                temp_prodcount += newprodcount;

                cur += newcount[0];
#if STRIDED
                newoffset[0] += newcount[0]*size; // TODO: Verify this (size)
#else
                newoffset[0] += newcount[0];
#endif
                data_offset += newprodcount;
              }
              data_chunks.push_back(std::make_pair(
                            temp_padd_data_, temp_prodcount));
            }
          }
          else if (NC_INT == vartype && this->output_labels_) {
            std::size_t temp_padd_max_row_ = count[0];
            // padd_max_row_ = count[0];
            // shared_ptr<int> temp_padd_label_
            //               = shared_ptr<int>(new int[padd_max_row_]
            //                                , array_deleter<int>());
            vector<int> temp_padd_label_(temp_padd_max_row_);

            LOG(INFO) << "PnetCDF padd max_row_ = " << temp_padd_max_row_;
            if (prodcount < chunksize) {
              LOG(INFO) << "reading PnetCDF label whole " << count[0];
#if STRIDED
              retval = ncmpi_get_vars_int_all(ncid, varid, &offset[0],
                &count[0], &stride[0], &temp_padd_label_[0]);
#else
              retval = ncmpi_get_vara_int_all(ncid, varid, &offset[0],
                 &count[0], &temp_padd_label_[0]);
              // retval = ncmpi_get_vara_int_all(ncid, varid, &offset[0],
              //  &count[0], &(this->padd_label_[0]));

#endif
              label_chunks.push_back(std::make_pair(
                                         temp_padd_label_, count[0]));
              // temp_padd_label_.resize(0);
              errcheck(retval);
            }
            else {
              vector<MPI_Offset> newoffset = offset;
              vector<MPI_Offset> newcount = count;
              MPI_Offset data_offset = 0;
              newcount[0] = 1;
              MPI_Offset newprodcount = prod(newcount);
              newcount[0] = chunksize/newprodcount;
              newprodcount = prod(newcount);
              if (newprodcount >= chunksize) {
                LOG(FATAL) << "newprodcount >= chunksize";
              }
              MPI_Offset cur = 0;
              //shared_ptr<int> chunk
              //            = shared_ptr<int>(new int[newprodcount]
              //                                , array_deleter<int>());
              vector<int> chunk(newprodcount);
              MPI_Offset temp_count = 0;
              while (cur < count[0]) {
                if (cur+newcount[0] > count[0]) {
                  newcount[0] = count[0]-cur;
                  newprodcount = prod(newcount);
                }
                LOG(INFO) << "reading label chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
                retval = ncmpi_get_vars_int_all(ncid, varid, &newoffset[0],
                    &newcount[0], &stride[0], &chunk[0]);
#else
                retval = ncmpi_get_vara_int_all(ncid, varid, &newoffset[0],
                    &newcount[0], &chunk[0]);
#endif
                errcheck(retval);
                memcpy(&temp_padd_label_[0] + data_offset, &chunk[0], newprodcount);
                temp_count += newcount[0];
                cur += newcount[0];
#if STRIDED
                newoffset[0] += newcount[0]*size;;
#else
                newoffset[0] += newcount[0];
#endif
                data_offset += newprodcount;
              }
              label_chunks.push_back(std::make_pair( temp_padd_label_
                                   , temp_count));
              //temp_padd_label_.resize(0);

            }
          }
          else {
            LOG(FATAL) << "unknown data type";
          }
      }
    }

    // Update data_, label_ here: ///////////////////////////////
    MPI_Offset temp_prodcount = 0, temp_labelcount = 0;
    vector<data_count_pair_type>::iterator ditr = data_chunks.begin();
    vector<label_count_pair_type>::iterator litr = label_chunks.begin();

    DLOG(INFO) << "DataChunk Size: " << data_chunks.size()
               << " LabelChunk Size: " << label_chunks.size()
               << " _________________";

    for(;ditr != data_chunks.end(); ditr++)
      temp_prodcount += (*ditr).second;

    for(;litr != label_chunks.end();litr++)
      temp_labelcount += (*litr).second;

    this->padd_data_char_count_ += temp_prodcount;
    this->padd_label_int_count_ += temp_labelcount;
    this->padd_max_row_ += temp_labelcount;

    DLOG(INFO) << "Temp_Prodcount val: " << this->padd_data_char_count_
               << " Temp_Labelcount val: " << this->padd_label_int_count_
               << " ,rank:" << comm_rank_;

    this->padd_data_.resize(0);
    this->padd_label_.resize(0);

    MPI_Offset temp_pcount = 0, temp_lcount = 0;
    row_mutex_->lock();

    for(ditr = data_chunks.begin()
        ; ditr != data_chunks.end(); ditr++) {
      // memcpy(this->padd_data_.get() + temp_pcount, (*ditr).first.get()
      //        , (*ditr).second);
      assert((*ditr).second == ((*ditr).first).size());
      // copy((*ditr).first.begin(), (*ditr).first.end(), back_inserter(this->padd_data_));
      copy((*ditr).first.begin(), (*ditr).first.end(), back_inserter(this->data_));
      temp_pcount += (*ditr).second;
      // release memory?
    }

    DLOG(INFO) << "_____LabelChunk_Size: " << label_chunks.size();

    for(litr = label_chunks.begin()
        ; litr != label_chunks.end(); litr++) {
      assert((*litr).second == ((*litr).first).size());
      // copy((*litr).first.begin(), (*litr).first.end(), back_inserter(this->padd_label_));
      copy((*litr).first.begin(), (*litr).first.end(), back_inserter(this->label_));
      temp_lcount += (*litr).second;
      this->max_row_ += (*litr).second;
      this->current_row_ = 0;
    }
    row_mutex_->unlock();

    retval = ncmpi_close(ncid);
    errcheck(retval);

    row_mutex_.reset(new boost::mutex());

    DLOG(INFO) << "Total Padd Data Char Count:" << padd_data_char_count_;
    DLOG(INFO) << "Total Padd Label Int Count:" << padd_label_int_count_;

    /*{
      const int batch_size = this->layer_param_.data_param().batch_size();
      Dtype label_sum = 0;
      for (int i=0; i<batch_size; i++) {
        label_sum += this->padd_label_[i];
      }
      LOG(INFO) << "Label Sum (Before AllReduce update): " << label_sum;
      // caffe::mpi::allreduce(label_sum);
      caffe::mpi::allreduce(&label_sum, 1, MPI_SUM, this->comm_ );
      // LOG(INFO) << "Padd Label Sum: " << label_sum;
      LOG(INFO) << "Label Sum (After AllReduce update): "
                << label_sum/this->comm_size_;
    }*/
    //}
  } else {
    LOG(INFO) << "No failed Rank detected";
  }

#else
  NO_PNETCDF;
#endif /*USE_PNETCDF*/
}
#endif /*CAFFE_FT*/
//////////////////////////////////////////////////////
template <typename Dtype>
size_t PnetCDFAllDataLayer<Dtype>::get_datum_size() {
  vector<int> top_shape = this->get_datum_shape();
  const size_t datum_channels = top_shape[1];
  const size_t datum_height = top_shape[2];
  const size_t datum_width = top_shape[3];
  return datum_channels*datum_height*datum_width;
}

template <typename Dtype>
vector<int> PnetCDFAllDataLayer<Dtype>::get_datum_shape() {
  // CHECK(this->data_);
  CHECK(this->datum_shape_.size());
  return this->datum_shape_;
}

template <typename Dtype>
vector<int> PnetCDFAllDataLayer<Dtype>::infer_blob_shape() {
  vector<int> top_shape = this->get_datum_shape();
  const int crop_size = this->transform_param_.crop_size();
  const int datum_height = top_shape[2];
  const int datum_width = top_shape[3];
  // Check dimensions.
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  top_shape[2] = (crop_size)? crop_size: datum_height;
  top_shape[3] = (crop_size)? crop_size: datum_width;
  return top_shape;
}

template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();

#ifdef CAFFE_FT
  CPUTimer data_timer;
  double original_data_load_time = 0;
  data_timer.Start();
#endif

  // Load the pnetcdf file into data_ and optionally label_
  load_pnetcdf_file_data(this->layer_param_.data_param().source());
#ifdef CAFFE_FT
  original_data_load_time += data_timer.MilliSeconds();
#endif

  row_mutex_.reset(new boost::mutex());

#ifdef CAFFE_FT
  //if(this->comm_rank_ == 0)
    LOG(INFO) << "Original_data_load_time: " << original_data_load_time << " ms. for_rank: " << this->comm_rank_;
#endif

  vector<int> top_shape = infer_blob_shape();
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

#ifdef CAFFE_FT
template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::DataLayerUpdate(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  this->comm_ = caffe::mpi::get_working_comm();
  this->comm_rank_ = caffe::mpi::comm_rank(this->comm_);
  this->comm_size_ = caffe::mpi::comm_size(this->comm_);

  const int batch_size = this->layer_param_.data_param().batch_size();

  CPUTimer data_timer;
  double data_update_time = 0, data_reload_time = 0;
  data_timer.Start();
  reload_pnetcdf_file_data(this->layer_param_.data_param().source());
  data_reload_time += data_timer.MilliSeconds();

  // CHECK(this->data_);
  // CHECK(this->padd_data_);
  // CHECK(this->label_);
  // CHECK(this->padd_label_);
  row_mutex_->lock();

  DLOG(INFO) << "Checking Data, padd_data, label, padd_label" ;

  DLOG(INFO) << "DataLayerUpdate: data_char_count: " << this->data_char_count_
             << " ,padd_data_char_count: " << this->padd_data_char_count_
             << " ,rank: " << this->comm_rank_;

  DLOG(INFO) << "After Padd Data copy to Data_____________" ;

  data_timer.Start();
  // this->data_.swap(this->padd_data_);
  //#### copy(this->padd_data_.begin(), this->padd_data_.end(), back_inserter(this->data_));
  //for(int i = 0; i < this->padd_data_.size(); ++i)
  //  this->data_.push_back(this->padd_data_[i]);
  /*
  data_timer.Start();
  vector<signed char> temp_data;
  for(int i = 0; i < this->data_.size(); ++i)
    temp_data.push_back(this->data_[i]);
  for(int i = 0; i < this->padd_data_.size(); ++i)
    temp_data.push_back(this->padd_data_[i]);
  */
  // copy(this->data_.begin(), this->data_.end(), back_inserter(temp_data));
  // copy(this->padd_data_.begin(), this->padd_data_.end(), back_inserter(temp_data));
  // this->data_.swap(temp_data);

  //this->label_.swap(this->padd_label_);
  // this->label_.swap(this->padd_label_);
  //#### copy(this->padd_label_.begin(), this->padd_label_.end(), back_inserter(this->label_));
  //for(int i = 0; i < this->padd_label_.size(); ++i)
  //  this->label_.push_back(this->padd_label_[i]);
  /*
  vector<int> temp_label;
  for(int i = 0; i < this->label_.size(); ++i)
    temp_label.push_back(this->label_[i]);
  for(int i = 0; i < this->padd_label_.size(); ++i)
    temp_label.push_back(this->padd_label_[i]);
  */
  // copy(this->label_.begin(), this->label_.end(), back_inserter(temp_label));
  // copy(this->padd_label_.begin(), this->padd_label_.end(), back_inserter(temp_label));
  // this->label_.swap(temp_label);

  DLOG(INFO) << "After Padd Label copy to Label" ;

  //#### this->max_row_ += this->padd_max_row_;
  // this->max_row_ = this->padd_max_row_;
  //#### this->current_row_ = 0;
  //#### this->data_char_count_ += this->padd_data_char_count_;
  //#### this->label_int_count_ += this->padd_label_int_count_;
  // this->data_char_count_ = this->padd_data_char_count_;
  // this->label_int_count_ = this->padd_label_int_count_;
  data_update_time += data_timer.MilliSeconds();

  DLOG(INFO) << "Total Data Char Count, after padd:" << this->data_char_count_;
  DLOG(INFO) << "Total Label Int Count, after padd:" << this->label_int_count_;
  this->padd_data_.resize(0);
  this->padd_label_.resize(0);
  this->padd_data_char_count_ = 0;
  this->padd_label_int_count_ = 0;
  this->padd_max_row_ = 0;
  // data_update_time += data_timer.MilliSeconds();

  row_mutex_->unlock();

  // if(this->comm_rank_ == 0) {
    LOG(INFO) << "Data_reload_time: " << data_reload_time << " ms. for_rank: "<< this->comm_rank_;
    LOG(INFO) << "Data_update_time: " << data_update_time << " ms. for_rank: "<< this->comm_rank_;

  // }

  row_mutex_.reset(new boost::mutex());
  vector<int> top_shape = infer_blob_shape();
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

}
#endif /*CAFFE_FT*/

// This function is called on prefetch thread
template<typename Dtype>
void PnetCDFAllDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

//##ifndef _OPENMP
  CHECK(this->transformed_data_.count());
//##endif

  vector<int> top_shape = get_datum_shape();
  size_t datum_size = get_datum_size();
  Datum masterDatum;
  masterDatum.set_channels(top_shape[1]);
  masterDatum.set_height(top_shape[2]);
  masterDatum.set_width(top_shape[3]);

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  top_shape = infer_blob_shape();
//##ifndef _OPENMP
  this->transformed_data_.Reshape(top_shape);
//##endif
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  // set up the Datum

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
//##ifdef _OPENMP
//#  #pragma omp parallel if (batch_size > 1)
//#  #pragma omp single nowait
//##endif
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    size_t row = (current_row_+item_id) % this->max_row_;
    size_t pnetcdf_offset = row * datum_size;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
//#    #pragma omp task firstprivate(masterDatum, top_label, item_id)
    {
      Datum datum = masterDatum;
      // datum.set_data(this->data_.get() + pnetcdf_offset, datum_size);
      datum.set_data(&(this->data_[0]) + pnetcdf_offset, datum_size);

      int offset = batch->data_.offset(item_id);
//##ifdef _OPENMP
//#      Blob<Dtype> tmp_data;
//#      tmp_data.Reshape(top_shape);
//#      tmp_data.set_cpu_data(top_data + offset);
//#      this->data_transformer_->Transform(datum, &tmp_data);
//##else
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
//##endif
      // Copy label.
      if (this->output_labels_) {
        //top_label[item_id] = this->label_.get()[row];
        top_label[item_id] = this->label_[row];
      }
    }
    trans_time += timer.MicroSeconds();
  }

  current_row_+=batch_size;

  timer.Stop();
  batch_timer.Stop();
#ifdef CAFFE_FT
  // if(this->comm_rank_ == 0)
    LOG(INFO) << "Prefetch_batch_time: " << batch_timer.MilliSeconds() << " ms. for_rank: " << this->comm_rank_;
#endif

  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
size_t PnetCDFAllDataLayer<Dtype>::next_row() {
  size_t row;
  row_mutex_->lock();
  row = current_row_++;
  current_row_ = current_row_ % this->max_row_;
  row_mutex_->unlock();
  return row;
}

INSTANTIATE_CLASS(PnetCDFAllDataLayer);
REGISTER_LAYER_CLASS(PnetCDFAllData);

}  // namespace caffe

