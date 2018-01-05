/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef USE_REMOTE_INDEX_SFTP
#ifndef CAFFE_UTIL_DB_REMOTE_INDEX_SFTP_HPP
#define CAFFE_UTIL_DB_REMOTE_INDEX_SFTP_HPP

#include <libssh/sftp.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <utility>
#include <vector>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <cstdlib>

#include "caffe/util/db.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/blocking_queue.hpp"

namespace caffe { namespace db {

class RemoteIndexSFTPEnv
{
  public:

  RemoteIndexSFTPEnv():
    total_timer(),
    timer()
  {
      index_buffer_size = 16*1024;
      buffer_size = 1024;
      buffer = new char[buffer_size];
      index_buffer = new char[index_buffer_size];
      current_index = 0;
      index_position = 0;
      block_position = 0;
      valid_=false;
      bytes_total = 0;
      total_timer.Start();
  }

  void get_next(string& key, string& value)
  {
    boost::lock_guard<boost::mutex> lck(mtx_);

    if(current_index >= image_index.size())
    {
      valid_=false;
      return;
    }
    //value.resize(image_index[current_index]);
    if(image_index[current_index]>buffer_size)
    {
      delete[] buffer;
      buffer_size = image_index[current_index];
      buffer = new char[buffer_size];
    }
    uint64_t res=0, total = 0;

    // timer.Start();
    res = sftp_read( block_file_, buffer, image_index[current_index]);
    total+=res;
    while(total != image_index[current_index])
    {
      res = sftp_read( block_file_, buffer+total, image_index[current_index]-total);
      total+=res;
    }
    bytes_total+= image_index[current_index];
    if(current_index > 0 && current_index % 1000 == 0)
    {
      DLOG(INFO) << "Current Index: " << current_index;
      LOG(INFO) << "Avg. "  << (((double)bytes_total)/total_timer.Seconds())/(1024*1024) << "MB/s Speed ";
      total_timer.Start();
      bytes_total = 0;
    }
    value = string(buffer, image_index[current_index]);
    //LOG(INFO) << total << "/" << image_index[current_index] << " "  << value.size() << " " << key_index[current_index];
    key = key_index[current_index++];//std::to_string(current_index++);
  }

  void put(string& key, string& value)
  {
    //index_stream_ << key << ":" << value.size() << ",";
    //block_stream_.write( value.c_str(), value.size());
    //block_stream_ << value; //.write( value.data(), value.size());
  }
  uint64_t helper_getline(char * buffer, string & key, char delim, uint64_t & pos, uint64_t max_size)
  {
    bool found = false;
    for(uint64_t i =0; i< max_size; i++)
    {
      if(buffer[i+pos] == delim)
      {
        string tmp(string(buffer+pos, i));
        key = "";
        if(tmp[0] == '\0') {
           LOG(INFO) << "NULL Character Found! ";
           key = string(tmp.begin() +1, tmp.end());
        }
        else
          key = tmp;
        //DLOG(INFO) << "KEY VAL: " << key;
        found = true;
        pos+=i+1;
        break;
      }
    }
    if(!found)
    {
      DLOG(INFO) << "Delimiter Not Found!!";
      memcpy(buffer,buffer+pos, max_size);
      return max_size;
    }
    return 0;
  }

  int verify_knownhost(ssh_session session)
  {
      int state, hlen;
      unsigned char *hash = NULL;
      char *hexa;
      char buf[10];
      state = ssh_is_server_known(session);
      hlen = ssh_get_pubkey_hash(session, &hash);
      if (hlen < 0)
          return -1;
      switch (state)
      {
          case SSH_SERVER_KNOWN_OK:
              break; /* ok */
          case SSH_SERVER_KNOWN_CHANGED:
              fprintf(stderr, "Host key for server changed: it is now:\n");
              ssh_print_hexa("Public key hash", hash, hlen);
              fprintf(stderr, "For security reasons, connection will be stopped\n");
              free(hash);
              return -1;
          case SSH_SERVER_FOUND_OTHER:
              fprintf(stderr, "The host key for this server was not found but an other"
              "type of key exists.\n");
              fprintf(stderr, "An attacker might change the default server key to"
              "confuse your client into thinking the key does not exist\n");
              free(hash);
              return -1;
          case SSH_SERVER_FILE_NOT_FOUND:
              fprintf(stderr, "Could not find known host file.\n");
              fprintf(stderr, "If you accept the host key here, the file will be"
              "automatically created.\n");
              /* fallback to SSH_SERVER_NOT_KNOWN behavior */
          case SSH_SERVER_NOT_KNOWN:
              hexa = ssh_get_hexa(hash, hlen);
              fprintf(stderr,"The server is unknown. Do you trust the host key?\n");
              fprintf(stderr, "Public key hash: %s\n", hexa);
              free(hexa);
              fprintf(stderr, "Trusting the remote host for now");
              /*
               * TODO: REVERT THIS
              if (fgets(buf, sizeof(buf), stdin) == NULL)
              {
                  free(hash);
                  return -1;
              }
              if (strncasecmp(buf, "yes", 3) != 0)
              {
                  free(hash);
              return -1;
              }
              if (ssh_write_knownhost(session) < 0)
              {
                  fprintf(stderr, "Error %s\n", strerror(errno));
                  free(hash);
                  return -1;
              }*/
              break;
          case SSH_SERVER_ERROR:
              fprintf(stderr, "Error %s", ssh_get_error(session));
              free(hash);
              return -1;
      }
      DLOG(INFO) << "VERIFY_KNOWN_HOST: ----";
      free(hash);
      return 0;
  }

  int authenticate_pubkey(ssh_session session)
  {
    int rc;
    rc = ssh_userauth_publickey_auto(session, NULL, NULL);
    if (rc == SSH_AUTH_ERROR)
    {
      fprintf(stderr, "Authentication failed: %s\n",
      ssh_get_error(session));
      return SSH_AUTH_ERROR;
    }
    return rc;
  }

  int sftp_create_session(ssh_session session, sftp_session &sftp)
  {
      int rc;
      sftp = sftp_new(session);
      if (sftp == NULL)
      {
          fprintf(stderr, "Error allocating SFTP session: %s\n",
          ssh_get_error(session));
          return SSH_ERROR;
      }
      rc = sftp_init(sftp);
      if (rc != SSH_OK)
      {
          fprintf(stderr, "Error initializing SFTP session: %s.\n",
          sftp_get_error(sftp));
          sftp_free(sftp);
          return rc;
      }
      return SSH_OK;
  }
  void open_session(const char * host, const char * user, const char * password)
  {
    int rc;
    db_ssh_session = ssh_new();
    if (db_ssh_session == NULL)
        exit(-1);
    ssh_options_set(db_ssh_session, SSH_OPTIONS_HOST, host);
    ssh_options_set(db_ssh_session, SSH_OPTIONS_USER, user);
    rc = ssh_connect(db_ssh_session);
    if (rc != SSH_OK)
    {
        fprintf(stderr, "Error connecting to localhost: %s\n",
        ssh_get_error(db_ssh_session));
        exit(-1);
    }

    if (verify_knownhost(db_ssh_session) < 0)
    {
        ssh_disconnect(db_ssh_session);
        ssh_free(db_ssh_session);
        exit(-1);
    }

    rc = authenticate_pubkey(db_ssh_session);
    if (rc != SSH_AUTH_SUCCESS)
    {
      rc = ssh_userauth_password(db_ssh_session, NULL, password);
      if (rc != SSH_AUTH_SUCCESS)
      {
        password = getpass("Password: ");
        rc = ssh_userauth_password(db_ssh_session, NULL, password);
      }
    }
    if (rc != SSH_AUTH_SUCCESS)
    {
        fprintf(stderr, "Error authenticating with password: %s\n",
            ssh_get_error(db_ssh_session));
        ssh_disconnect(db_ssh_session);
        ssh_free(db_ssh_session);
        exit(-1);
    }

    sftp_create_session(db_ssh_session, db_sftp_session);
  }

  int open(const string &source, int sftp_mode, int mode, const string &server, const string &username, const string &password)
  {
    open_session(server.c_str(), username.c_str(), password.c_str());
    string index = source + "/index";
    string block = source + "/db";
    string key, int_string;
    index_file_ = sftp_open(db_sftp_session, index.c_str(), sftp_mode, 0);
    block_file_ = sftp_open(db_sftp_session, block.c_str(), sftp_mode, 0);
    uint64_t value = 0;
    valid_=true;
    uint64_t nbytes = 0;
    uint64_t res = 0;
    uint64_t pos = 0 ;
    uint64_t offset = 0;
    uint64_t read_offset = 0;
    bool break2= false;
    char last_char = ',';
    if(sftp_mode == O_RDONLY)
    {
      //while(!index_stream_.eof())
      while(1)
      {
          //index_stream.ignore(numeric_limits<streamsize>::max(),':');
          read_offset = offset;
          nbytes = sftp_read(index_file_, index_buffer+read_offset, index_buffer_size-(read_offset));
          read_offset += nbytes;
          while(nbytes != 0 && (index_buffer_size-(read_offset)) > 0)
          {
            nbytes = sftp_read(index_file_, index_buffer+read_offset, index_buffer_size-(read_offset));
            read_offset += nbytes;
          }
          if(read_offset == 0) {
            LOG(INFO) << "Image Index Key Size: " << key_index.size();
            LOG(INFO) << "Image Index Size: " << image_index.size();
            return 0;
          }
          else
            break2 = false;

          pos = 0;

        offset=0;
        while(1)
        {
          // if(!break2)
          // {
          if(last_char == ',') {
            // key = "";
            res = helper_getline(index_buffer,key,':', pos
                , ((read_offset-pos)? (read_offset - pos):1));
            //LOG(INFO) << pos << " Key " << key << " " << res;
            if( res) // || pos == read_offset )
            {
              // LOG(INFO) << "RES value: " << res;
              // if(!res)
              //   key_index.push_back(key);
              offset = res;
              break;
            }

            key_index.push_back(key);
            last_char = ':';
          // }
          // if(read_offset-pos == 0)
          // {
          //   offset=0;
          //   break;
          // }
          }
          if( last_char == ':') {
            // int_string = "";
            res = helper_getline(index_buffer,int_string,',', pos
              , ((read_offset-pos) ? (read_offset - pos):1));
            if(res) //|| pos == read_offset )
            {
              offset = res;
              // if(!res)
              // {
              //   value = std::strtoull(int_string.c_str(),NULL,0);
              //   //LOG(INFO) << pos << " Val2 " << value << "   " << read_offset;
              //   image_index.push_back(value);
              //   break;
              // }
              // break2=true;
              break;
            }
          // else
          //   break2=false;
            value = std::strtoull(int_string.c_str(),NULL,0);
            //LOG(INFO) << pos << " Val " << value << "   " << read_offset;
            image_index.push_back(value);
            last_char = ',';
          }

          if(read_offset-pos == 0)
          {
            offset=0;
            break;
          }
          // offset=0;
        }
      }
    }
    LOG(INFO) << "Image Index Key Size: " << key_index.size();
    LOG(INFO) << "Image Index Size: " << image_index.size();
    return 0;
  }
  void close()
  {
     sftp_close(index_file_);
     sftp_close(block_file_);
  }
  /*void seek(string& key, string& value)
  {
    current_index = std::stoull(key);
    value.resize(image_index[current_index]);
    block_stream_.seek();
    block_stream_.read( &value, image_index[current_index]);
    key = std::to_string(current_index++);
    if(current_index > image_index.size())
    {
      current_index = 0;
    }

  }*/

  bool valid(){ return valid_; };

  void reset() { valid_=true; sftp_seek(index_file_, 0); sftp_seek(block_file_, 0); current_index =0;  }

  protected:
  boost::mutex mtx_;

  private:
  ssh_session db_ssh_session;
  sftp_session db_sftp_session;

  // Master List of image sizes
  vector<uint64_t> image_index;
  vector<string> key_index;
  bool valid_;
  char * buffer;
  char * index_buffer;
  uint64_t buffer_size;
  uint64_t index_buffer_size;
  sftp_file index_file_;
  sftp_file block_file_;
  uint64_t current_index;
  uint64_t index_position;
  uint64_t block_position;
  uint64_t bytes_total;
  Timer timer;
  Timer total_timer;
};

class RemoteIndexSFTPCursor : public Cursor {
 public:
  explicit RemoteIndexSFTPCursor(RemoteIndexSFTPEnv * env)
    : env_(env), valid_(false) {
    SeekToFirst();
  }
  virtual ~RemoteIndexSFTPCursor() {
    // env_->close();
  }
  virtual void SeekToFirst() { env_->reset(); env_->get_next(key_, value_); }
  virtual void Next() { env_->get_next(key_, value_); }
  //virtual void SeekToImage() { env->seek(key_, value_); }
  //virtual void EntryCount() { return env->EntryCount(); }
  virtual string key() {
    return key_;
  }
  virtual string value() {
    return value_;
  }
  virtual std::pair<void*, size_t> valuePointer() {
    return std::make_pair((void*)value_.c_str(), value_.size());
  }

  virtual bool valid() { return env_->valid(); }

 private:
  RemoteIndexSFTPEnv * env_;
  string key_, value_;
  bool valid_;
};

class RemoteIndexSFTPTransaction : public Transaction {
 public:
  explicit RemoteIndexSFTPTransaction(RemoteIndexSFTPEnv * env):
  env_(env)
  { }
  virtual void Put(const string& key, const string& value);
  virtual void Commit();

 private:
  vector<string> keys, values;
  RemoteIndexSFTPEnv * env_;

  DISABLE_COPY_AND_ASSIGN(RemoteIndexSFTPTransaction);
};

class RemoteIndexSFTP : public DB {
 public:
  RemoteIndexSFTP() { }
  virtual ~RemoteIndexSFTP() { Close(); }
  virtual void Open(const string& source, Mode mode, const LayerParameter * param);
  virtual void Close() { env_->close();
  }
  virtual RemoteIndexSFTPCursor* NewCursor();
  virtual RemoteIndexSFTPTransaction* NewTransaction();

 private:
  RemoteIndexSFTPEnv * env_;
};

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_LMDB_HPP
#endif  // USE_LMDB
