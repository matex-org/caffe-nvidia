#include <stdint.h>
#include <string.h>

#include <vector>

#include <netcdf.h>

#include <boost/thread.hpp>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/netcdf_bcast_data_layer.hpp"
#include "caffe/mpi.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
NetCDFBcastDataLayer<Dtype>::NetCDFBcastDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    current_row_(0),
    max_row_(0),
    datum_shape_(),
    data_(),
    label_(),
    row_mutex_(),
    comm_(),
    comm_rank_(),
    comm_size_() {
  comm_ = caffe::mpi::comm_dup();
  comm_rank_ = caffe::mpi::comm_rank(comm_);
  comm_size_ = caffe::mpi::comm_size(comm_);
}

template <typename Dtype>
NetCDFBcastDataLayer<Dtype>::~NetCDFBcastDataLayer() {
  this->StopInternalThread();
}

static void errcheck(int retval) {
  if (NC_NOERR != retval) {
    LOG(FATAL) << "netcdf error: " << nc_strerror(retval);
  }
}

template <typename Dtype>
inline static Dtype prod(vector<Dtype> vec) {
  Dtype val = 1;
  for (size_t i=0; i<vec.size(); i++) {
    val *= vec[i];
  }
  return val;
}

template <typename Dtype>
void NetCDFBcastDataLayer<Dtype>::load_netcdf_file_data(const string& filename) {
  LOG(INFO) << "Loading NetCDF file: " << filename;

  int rank = comm_rank_;
  int retval;
  int ncid;
  int ndims;
  int nvars;
  int ngatts;
  int unlimdim;
  size_t total;
  size_t remain;
  size_t start;
  size_t stop;
  CPUTimer timer;
  size_t chunksize = 2147483647L;
  unsigned long data_size = 0;
  unsigned long label_size = 0;

  timer.Start();

  if (0 == rank) {
    retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    errcheck(retval);

    retval = nc_inq(ncid, &ndims, &nvars, &ngatts, &unlimdim);
    errcheck(retval);

    retval = nc_inq_dimlen(ncid, unlimdim, &total);
    errcheck(retval);

    start = 0;
    stop = total;

    DLOG(INFO) << "ncid " << ncid;
    DLOG(INFO) << "ndims " << ndims;
    DLOG(INFO) << "nvars " << nvars;
    DLOG(INFO) << "ngatts " << ngatts;
    DLOG(INFO) << "unlimdim " << unlimdim;
    LOG(INFO) << "total images " << total;
    LOG(INFO) << "remain " << remain;
    LOG(INFO) << "start " << start;
    LOG(INFO) << "stop " << stop;

    for (int varid = 0; varid < nvars; varid++) {
      int vartype;
      int varndims;
      vector<int> vardimids;
      vector<size_t> count;
      vector<size_t> offset;
      vector<size_t> stride;
      size_t prodcount;

      retval = nc_inq_vartype(ncid, varid, &vartype);
      errcheck(retval);

      retval = nc_inq_varndims(ncid, varid, &varndims);
      errcheck(retval);

      vardimids.resize(varndims);
      count.resize(varndims);
      offset.resize(varndims);
      stride.resize(varndims);

      retval = nc_inq_vardimid(ncid, varid, &vardimids[0]);
      errcheck(retval);

      for (int i = 0; i < varndims; i++) {
        retval = nc_inq_dimlen(ncid, vardimids[i], &count[i]);
        errcheck(retval);
        offset[i] = 0;
        stride[i] = 1;
        if (count[i] > chunksize) {
          LOG(FATAL) << "dimension is too large for Blob";
        }
      }
      // MPI-IO can only read 2GB chunks due to "int" interface for indices
      count[0] = stop-start;
      offset[0] = start;
      prodcount = prod(count);

      LOG(INFO) << "prodcount " << prodcount;
      LOG(INFO) << "chunksize " << chunksize;

      if (NC_BYTE == vartype) {
        this->data_ = shared_ptr<signed char>(new signed char[prodcount]);
        data_size = prodcount;
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
          LOG(INFO) << "reading NetCDF data whole " << count[0];
          LOG(INFO) << "offset={"<<offset[0]<<","<<offset[1]<<","<<offset[2]<<","<<offset[3]<<"}";
          LOG(INFO) << "count={"<<count[0]<<","<<count[1]<<","<<count[2]<<","<<count[3]<<"}";
          retval = nc_get_vara_schar(ncid, varid, &offset[0],
              &count[0], this->data_.get());
          errcheck(retval);
        }
        else {
          vector<size_t> newoffset = offset;
          vector<size_t> newcount = count;
          size_t data_offset = 0;
          newcount[0] = 1;
          size_t newprodcount = prod(newcount);
          newcount[0] = chunksize/newprodcount;
          newprodcount = prod(newcount);
          if (newprodcount >= chunksize) {
            LOG(FATAL) << "newprodcount >= chunksize";
          }
          size_t cur = 0;
          shared_ptr<signed char> chunk = shared_ptr<signed char>(
              new signed char[newprodcount]);
          while (cur < count[0]) {
            if (cur+newcount[0] > count[0]) {
              newcount[0] = count[0]-cur;
              newprodcount = prod(newcount);
            }
            LOG(INFO) << "reading data chunk " << cur << " ... " << cur+newcount[0];
            retval = nc_get_vara_schar(ncid, varid, &newoffset[0],
                &newcount[0], chunk.get());
            errcheck(retval);
            memcpy(this->data_.get() + data_offset, chunk.get(), newprodcount);
            cur += newcount[0];
            newoffset[0] += newcount[0];
            data_offset += newprodcount;
          }
        }
      }
      else if (NC_INT == vartype && this->output_labels_) {
        max_row_ = count[0];
        this->label_ = shared_ptr<int>(new int[max_row_]);
        label_size = max_row_;
        LOG(INFO) << "NetCDF max_row_ = " << max_row_;
        if (prodcount < chunksize) {
          LOG(INFO) << "reading NetCDF label whole " << count[0];
          retval = nc_get_vara_int(ncid, varid, &offset[0],
              &count[0], this->label_.get());
          errcheck(retval);
        }
        else {
          vector<size_t> newoffset = offset;
          vector<size_t> newcount = count;
          size_t data_offset = 0;
          newcount[0] = 1;
          size_t newprodcount = prod(newcount);
          newcount[0] = chunksize/newprodcount;
          newprodcount = prod(newcount);
          if (newprodcount >= chunksize) {
            LOG(FATAL) << "newprodcount >= chunksize";
          }
          size_t cur = 0;
          shared_ptr<int> chunk = shared_ptr<int>(new int[newprodcount]);
          while (cur < count[0]) {
            if (cur+newcount[0] > count[0]) {
              newcount[0] = count[0]-cur;
              newprodcount = prod(newcount);
            }
            LOG(INFO) << "reading label chunk " << cur << " ... " << cur+newcount[0];
            retval = nc_get_vara_int(ncid, varid, &newoffset[0],
                &newcount[0], chunk.get());
            errcheck(retval);
            memcpy(this->label_.get() + data_offset, chunk.get(), newprodcount);
            cur += newcount[0];
            newoffset[0] += newcount[0];
            data_offset += newprodcount;
          }
        }
      }
      else {
        LOG(FATAL) << "unknown data type";
      }
    }

    retval = nc_close(ncid);
    errcheck(retval);

  }

  /* bcast the data to all ranks */
  caffe::mpi::bcast(max_row_, 0, comm_);
  caffe::mpi::bcast(data_size, 0, comm_);
  caffe::mpi::bcast(label_size, 0, comm_);
  if (0 != rank) {
    this->data_ = shared_ptr<signed char>(new signed char[data_size]);
    this->label_ = shared_ptr<int>(new int[label_size]);
    datum_shape_.resize(4);
  }
  caffe::mpi::bcast(datum_shape_, 0, comm_);

  LOG(INFO) << "datum_shape_[0] " << datum_shape_[0];
  LOG(INFO) << "datum_shape_[1] " << datum_shape_[1];
  LOG(INFO) << "datum_shape_[2] " << datum_shape_[2];
  LOG(INFO) << "datum_shape_[3] " << datum_shape_[3];

  /* how many elements do we have to bcast? avoid 2GB limits */
  int chunk = 2147483647; /* int32 limit */
  if (data_size > chunk) {
    LOG(INFO) << "too many data elements to bcast at once";
    int count = chunk;
    unsigned long offset = 0;
    while (offset < data_size) {
      if (offset+chunksize > data_size) {
        count = data_size - offset;
      }
      LOG(INFO) << "bcast data chunk " << offset << " ... " << offset+count;
      caffe::mpi::bcast(this->data_.get()+offset, count, 0, comm_);
      offset += chunksize;
    }
  }
  else {
    LOG(INFO) << "broadcasting data whole " << data_size;
    caffe::mpi::bcast(this->data_.get(), data_size, 0, comm_);
  }
  if (label_size > chunksize) {
    LOG(FATAL) << "too many label elements to bcast at once";
  }
  else {
    LOG(INFO) << "broadcasting label whole " << label_size;
    caffe::mpi::bcast(this->label_.get(), label_size, 0, comm_);
  }

#if 0
  {
    const int batch_size = this->layer_param_.data_param().batch_size();
    Dtype label_sum = 0;
    for (int i=0; i<batch_size; i++) {
      label_sum += *(this->label_.get()+i);
    }
    caffe::mpi::allreduce(label_sum);
    LOG(INFO) << "Label Sum: " << label_sum;
  }
#endif

  LOG(INFO) << "Data load time: " << timer.MilliSeconds() << " ms.";
}

template <typename Dtype>
size_t NetCDFBcastDataLayer<Dtype>::get_datum_size() {
  vector<int> top_shape = this->get_datum_shape();
  const size_t datum_channels = top_shape[1];
  const size_t datum_height = top_shape[2];
  const size_t datum_width = top_shape[3];
  return datum_channels*datum_height*datum_width;
}

template <typename Dtype>
vector<int> NetCDFBcastDataLayer<Dtype>::get_datum_shape() {
  CHECK(this->data_);
  CHECK(this->datum_shape_.size());
  return this->datum_shape_;
}

template <typename Dtype>
vector<int> NetCDFBcastDataLayer<Dtype>::infer_blob_shape() {
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
void NetCDFBcastDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();

  // Load the netcdf file into data_ and optionally label_
  load_netcdf_file_data(this->layer_param_.data_param().source());

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

// This function is called on prefetch thread
template<typename Dtype>
void NetCDFBcastDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  CHECK(this->transformed_data_.count());

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
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  // set up the Datum

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    size_t row = (current_row_+item_id) % this->max_row_;
    size_t netcdf_offset = row * datum_size;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    {
      Datum datum = masterDatum;
      datum.set_data(this->data_.get() + netcdf_offset, datum_size);

      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
      // Copy label.
      if (this->output_labels_) {
        top_label[item_id] = this->label_.get()[row];
      }
    }
    trans_time += timer.MicroSeconds();
  }

  current_row_+=batch_size;

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
size_t NetCDFBcastDataLayer<Dtype>::next_row() {
  size_t row;
  row_mutex_->lock();
  row = current_row_++;
  current_row_ = current_row_ % this->max_row_;
  row_mutex_->unlock();
  return row;
}

INSTANTIATE_CLASS(NetCDFBcastDataLayer);
REGISTER_LAYER_CLASS(NetCDFBcastData);

}  // namespace caffe

