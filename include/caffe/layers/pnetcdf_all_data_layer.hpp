#ifndef PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_
#define PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_

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
class PnetCDFAllDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PnetCDFAllDataLayer(const LayerParameter& param);
  virtual ~PnetCDFAllDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "PnetCDFData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_pnetcdf_file_data(const string& filename);
  #ifdef CAFFE_FT
  virtual void reload_pnetcdf_file_data(const string& filename);
  virtual std::tuple<int, bool> fix_comm_error(MPI_Comm* comm, float* x);
  virtual void DataLayerUpdate(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  #endif
  virtual void load_batch(Batch<Dtype>* batch);
  virtual vector<int> get_datum_shape();
  virtual size_t get_datum_size();
  virtual vector<int> infer_blob_shape();
  virtual size_t next_row();

  // std::size_t data_char_count_;
  // std::size_t label_int_count_;
  MPI_Offset data_char_count_;
  MPI_Offset label_int_count_;
  size_t current_row_;
  size_t max_row_;
  vector<int> datum_shape_;
  // shared_ptr<signed char> data_;
  // shared_ptr<int[]> label_;
  vector<signed char> data_;
  vector<int> label_;
  shared_ptr<boost::mutex> row_mutex_;
  MPI_Comm comm_;
  int comm_rank_;
  int comm_size_;
#ifdef CAFFE_FT
  size_t padd_max_row_;
  // size_t padd_data_char_count_;
  // size_t padd_label_int_count_;
  MPI_Offset padd_data_char_count_;
  MPI_Offset padd_label_int_count_;
  // shared_ptr<signed char> padd_data_;
  // shared_ptr<int[]> padd_label_;
  vector<signed char> padd_data_;
  vector<int> padd_label_;
  #ifdef USE_MPI
  int error_code_;
  #endif
private:
  void add_remaining(int rank, MPI_Offset remain
        , MPI_Offset* start, MPI_Offset* stop) {
            if(rank < remain) {
                *start += rank;
                *stop += rank + 1;
            } else {
                *start += remain;
                *stop += remain;
            }
        }

  template< typename T >
  struct array_deleter
  {
    void operator ()( T const * p){
      delete[] p;
    }
  };

/* void add_remaining(int rank, MPI_Offset remain,
        , MPI_Offset* start, MPI_Offset* stop) {

        } */
#endif /*CAFFE_FT*/
};

}  // namespace caffe

#endif  // PNETCDF_ALL_CAFFE_DATA_LAYER_HPP_

