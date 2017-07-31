#ifdef USE_LEVELDB
#include "caffe/util/db_leveldb.hpp"

#include <string>

namespace caffe { namespace db {

#ifdef USE_DEEPMEM
void LevelDB::Open(const string& source, Mode mode, const LayerParameter * param) {
#else
void LevelDB::Open(const string& source, Mode mode) {
#endif
  leveldb::Options options;
  options.block_size = 65536;
  options.write_buffer_size = 268435456;
  options.max_open_files = 100;
  options.error_if_exists = mode == NEW;
  options.create_if_missing = mode != READ;
  leveldb::Status status = leveldb::DB::Open(options, source, &db_);
  CHECK(status.ok()) << "Failed to open leveldb " << source
                     << std::endl << status.ToString();
  LOG(INFO) << "Opened leveldb " << source;
}

}  // namespace db
}  // namespace caffe
#endif  // USE_LEVELDB
