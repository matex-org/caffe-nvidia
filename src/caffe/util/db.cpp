#include "caffe/util/db.hpp"
#include "caffe/util/db_leveldb.hpp"
#include "caffe/util/db_lmdb.hpp"
#ifdef USE_DEEPMEM
#include "caffe/util/db_remote_index.hpp"
#ifdef USE_REMOTE_INDEX_SFTP
#include "caffe/util/db_remote_index_sftp.hpp"
#endif
#endif

#include <string>

namespace caffe { namespace db {

DB* GetDB(DataParameter::DB backend) {
  switch (backend) {
#ifdef USE_LEVELDB
  case DataParameter_DB_LEVELDB:
    return new LevelDB();
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  case DataParameter_DB_LMDB:
    return new LMDB();
#endif  // USE_LMDB
#ifdef USE_DEEPMEM
#ifdef USE_REMOTE_INDEX
      case DataParameter_DB_REMOTE_INDEX:
        return new RemoteIndex();
#endif
#ifdef USE_REMOTE_INDEX_SFTP
          case DataParameter_DB_REMOTE_INDEX_SFTP:
            return new RemoteIndexSFTP();
#endif
#endif
  default:
    LOG(FATAL) << "Unknown database backend";
    return NULL;
  }
}

DB* GetDB(const string& backend) {
#ifdef USE_LEVELDB
  if (backend == "leveldb") {
    return new LevelDB();
  }
#endif  // USE_LEVELDB
#ifdef USE_LMDB
  if (backend == "lmdb") {
    return new LMDB();
  }
#endif  // USE_LMDB
#ifdef USE_DEEPMEM
#ifdef USE_REMOTE_INDEX
    if (backend == "remote_index") {
          return new RemoteIndex();
            }
#endif
#ifdef USE_REMOTE_INDEX_SFTP
      if (backend == "remote_index_sftp") {
            return new RemoteIndexSFTP();
              }
#endif
#endif
  LOG(FATAL) << "Unknown database backend";
  return NULL;
}

}  // namespace db
}  // namespace caffe
