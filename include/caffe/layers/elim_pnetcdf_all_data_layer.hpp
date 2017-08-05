#ifndef ELIM_PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_
#define ELIM_PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/mpi.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

#define NO_PNETCDF LOG(FATAL) << "USE_PNETCDF not enabled in Makefile"

namespace caffe {

template <typename Dtype>
class ElimPnetCDFAllDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ElimPnetCDFAllDataLayer(const LayerParameter& param);
  virtual ~ElimPnetCDFAllDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "ElimPnetCDFData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual int BatchSize() const { return this->layer_param_.data_param().batch_size(); }
  virtual size_t MaxRow() const { return max_row_; }
  virtual int* Mask() { return mask_; }
  virtual size_t* Index() { return index_prefetch_current_; }

  using BasePrefetchingDataLayer<Dtype>::PREFETCH_COUNT;

 protected:
  virtual void load_pnetcdf_file_data(const string& filename);
  virtual void load_batch(Batch<Dtype>* batch);
  virtual vector<int> get_datum_shape();
  virtual size_t get_datum_size();
  virtual vector<int> infer_blob_shape();
  virtual size_t next_row();

  size_t current_row_;
  size_t max_row_;
  vector<int> datum_shape_;
  shared_ptr<signed char> data_;
  shared_ptr<int> label_;
  int *mask_;
  size_t * index_prefetch_current_;
  size_t * index_prefetch_[PREFETCH_COUNT];
  BlockingQueue<size_t*> index_prefetch_free_;
  BlockingQueue<size_t*> index_prefetch_full_;
  shared_ptr<boost::mutex> row_mutex_;
  MPI_Comm comm_;
  int comm_rank_;
  int comm_size_;
};

}  // namespace caffe

#endif  // ELIM_PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_

