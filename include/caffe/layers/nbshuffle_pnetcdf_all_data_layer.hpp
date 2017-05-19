#ifndef NBSHUFFLE_PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_
#define NBSHUFFLE_PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_

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
class NBShufflePnetCDFAllDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit NBShufflePnetCDFAllDataLayer(const LayerParameter& param);
  virtual ~NBShufflePnetCDFAllDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "PnetCDFData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  friend class Shuffler;

 protected:
  class Shuffler;

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
  signed char *one_data_;
  int *one_label_;
  int dest_;
  int source_;
  shared_ptr<boost::mutex> row_mutex_;
  MPI_Comm comm_;
  int comm_rank_;
  int comm_size_;
  bool ignore_;
  Shuffler *shuffler_;
  BlockingQueue<size_t> queue_;
};

}  // namespace caffe

#endif  // NBSHUFFLE_PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_

