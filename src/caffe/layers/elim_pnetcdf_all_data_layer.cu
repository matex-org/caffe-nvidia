#include <vector>

#include "caffe/layers/elim_pnetcdf_all_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void ElimPnetCDFAllDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BasePrefetchingDataLayer<Dtype>::Forward_gpu(bottom, top);
  index_prefetch_free_.push(index_prefetch_current_);
  index_prefetch_current_ = index_prefetch_full_.pop("Elim Data layer prefetch queue empty");
}

INSTANTIATE_LAYER_GPU_FORWARD(ElimPnetCDFAllDataLayer);

}  // namespace caffe
