#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#ifdef USE_DEEPMEM
// from Intel's version
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#endif

namespace caffe {

// #ifdef 0
#ifdef USE_DEEPMEM
// Intel's counter part here:

class RandNumbers {
 public:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  int operator()(int n) {
    CHECK_GT(n, 0);
    return GetNextNumber() % n;
  }

  virtual uint32_t GetNextNumber() = 0;
};

class GenRandNumbers: public RandNumbers {
 public:
  void Init() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  }
  void Reset() { rng_.reset(); }
  virtual uint32_t GetNextNumber() {
    CHECK(rng_);
    caffe::rng_t* rng = static_cast<caffe::rng_t*>(rng_->generator());
    return (*rng)();
  }
 private:
  shared_ptr<Caffe::RNG> rng_;
};

class PreclcRandomNumbers: public RandNumbers {
 public:
  void FillRandomNumbers(int num_count, RandNumbers& rand_gen) {
    for (int i = 0; i < num_count; i++)
      random_numbers.push(rand_gen.GetNextNumber());
  }

  virtual uint32_t GetNextNumber() {
    CHECK(!random_numbers.empty());
    uint32_t num = random_numbers.front();
    random_numbers.pop();
    return num;
  }
 private:
  std::queue<uint32_t> random_numbers;
};
#endif
// #endif

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

#ifndef CPU_ONLY
  void TransformGPU(int N, int C, int H, int W,
              const Dtype *in, Dtype *out, int *);
#endif
  void Copy(const Datum& datum, Dtype *data);
  void Copy(const cv::Mat& datum, Dtype *data);
  void CopyPtrEntry(string* str, Dtype* transformed_ptr,
                    bool output_labels, Dtype *label,
                    BlockingQueue<string*>* free);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param rand1
   *    Random value (0,RAND_MAX+1]
   * @param rand2
   *    Random value (0,RAND_MAX+1]
   * @param rand3
   *    Random value (0,RAND_MAX+1]
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
    void TransformPtrEntry(string* str, Dtype* transformed_ptr,
                           int rand1, int rand2, int rand3,
                           bool output_labels, Dtype *label,
                           BlockingQueue<string*>* free);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   * @param rand1
   *    Random value (0,RAND_MAX+1]
   * @param rand2
   *    Random value (0,RAND_MAX+1]
   * @param rand3
   *    Random value (0,RAND_MAX+1]
   */
  void TransformPtr(const cv::Mat& cv_img, Dtype* transformed_ptr,
                    int rand1, int rand2, int rand3);
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum, bool use_gpu = false);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img, bool use_gpu = false);
#endif  // USE_OPENCV

 protected:
  void TransformGPU(const Datum& datum, Dtype* transformed_data);
  void Transform(const Datum& datum, Dtype* transformed_data);
  // Tranformation parameters
  TransformationParameter param_;
  void TransformPtrInt(Datum* datum, Dtype* transformed_data,
                       int rand1, int rand2, int rand3);

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  Dtype *mean_values_gpu_ptr_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
