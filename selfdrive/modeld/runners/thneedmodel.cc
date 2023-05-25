#include "selfdrive/modeld/runners/thneedmodel.h"

#include <cassert>

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cstring> // For std::memcpy
#include <filesystem>

int accumulateDatas = 100; // 100 frames
int waitRecovery = 10; // 10s
int collectData = 1; // 1:true, 0:false
int readyToSave = 1; // 1:true, 0:false
long finishStamp = 0; 

const std::string LOGROOT = "/data/openpilot_log";
namespace fst = std::filesystem;

std::string SESSION;
std::string FOLDER;

size_t DESIRE_SIZE;
size_t TRAFFIC_SIZE;
size_t FEATURE_SIZE;
size_t OUTPUT_SIZE;
size_t FILE_SIZE;
size_t ImgSize;

std::shared_ptr<std::vector<char>> IMGBUFFER;
std::shared_ptr<std::vector<char>> FILEBUFFER;

struct CallbackData {
    std::shared_ptr<std::vector<char>> img_buffer;
    std::shared_ptr<std::vector<char>> file_buffer;
    size_t file_size;
    size_t files_written;
    int max_files;

    CallbackData(std::shared_ptr<std::vector<char>> i_buf,
		    std::shared_ptr<std::vector<char>> f_buf, 
		    size_t size, 
		    size_t written, 
		    int max)
    : img_buffer(i_buf), file_buffer(f_buf), file_size(size), files_written(written), max_files(max)
    {}
};

CallbackData* DATA;

int read_config(const std::string &filename) {
    //std::filesystem::path current_path = std::filesystem::current_path();
    //std::cerr << current_path;
    std::ifstream ifs;
    std::string str;

    ifs.open(filename);
    
    if (!ifs.is_open()) {
	std::cerr << "Failed to open file.\n";
        return 1; // EXIT_FAILURE
    }
    // Read the entire file into the string
    str.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    ifs.close();

    // Convert the string to an integer and return it
    try {
        return std::stoi(str);
    } catch(const std::invalid_argument& e) {
        std::cerr << "Invalid argument: " << e.what() << '\n';
        return 1; // EXIT_FAILURE
    } catch(const std::out_of_range& e) {
        std::cerr << "Out of range: " << e.what() << '\n';
        return 1; // EXIT_FAILURE
    }
}


size_t save_to_buffer(const float* src, size_t current_offset, size_t file_size) {

    size_t offset = DATA->files_written * FILE_SIZE + current_offset;
    // Convert float* to char* for copying
    const char* src_char = reinterpret_cast<const char*>(src);
    std::memcpy(DATA->file_buffer->data() + offset, src_char, file_size);
    return current_offset + file_size;
}

void CL_CALLBACK write_complete_callback(cl_event event, cl_int status, void *user_data) {
    cl_int err;
    
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    SESSION = std::to_string(ms);
    FOLDER = LOGROOT + "/" + SESSION;
    
    readyToSave = 0;
    fst::create_directory(FOLDER);

    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to complete writing data to file (" << status << ")" << std::endl;
        return;
    }
    auto data = static_cast<CallbackData*>(user_data);
    auto img_buffer = data->img_buffer;
    auto file_buffer = data->file_buffer;

    const std::string& img_path = FOLDER + "/img_inputs.bin";
    std::ofstream output_img(img_path, std::ios::binary);
    if (!output_img.is_open()) {
        std::cerr << "Error: Failed to open file \"" << img_path << "\"" << std::endl;
        return;
    }
    output_img.write(img_buffer->data(), img_buffer->size());
    output_img.close();

    const std::string& file_path = FOLDER + "/files.bin";
    std::ofstream output_file(file_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error: Failed to open file \"" << file_path << "\"" << std::endl;
        return;
    }
    output_file.write(file_buffer->data(), file_buffer->size());
    output_file.close();

    // Release event object
    err = clReleaseEvent(event);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to release read event (" << err << ")" << std::endl;
        return;
    }
    data->files_written=0;
    readyToSave = 1;
    finishStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}


bool save_clmem_to_file(const cl_mem cl_mem_obj, cl_context context, cl_command_queue command_queue, bool finish_this_cycle) {
    cl_int err;

    size_t offset;
    // Write to the next section of the buffer
    if (finish_this_cycle) {
    offset = DATA->files_written * 2 * ImgSize + ImgSize;
    }else{
    offset = DATA->files_written * 2 * ImgSize;
    }
    //std::cerr << "offset : " << offset << std::endl;
    
    // Read cl_mem_obj into buffer
    cl_event read_event;
    //err = clEnqueueReadBuffer(command_queue, cl_mem_obj, CL_FALSE, offset, ImgSize, DATA->buffer->data() + offset, 0, nullptr, &read_event);
    err = clEnqueueReadBuffer(command_queue, cl_mem_obj, CL_FALSE, 0, ImgSize, reinterpret_cast<void*>(DATA->img_buffer->data() + offset), 0, nullptr, &read_event);

    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to read cl_mem_obj (" << err << ")" << std::endl;
        return false;
    }
    if (finish_this_cycle){
    DATA->files_written++;
    }
    // Set callback function to write to file after buffer is full
    if (DATA->files_written >= DATA->max_files and readyToSave == 1) {
        err = clSetEventCallback(read_event, CL_COMPLETE, &write_complete_callback, DATA);
        if (err != CL_SUCCESS) {
            std::cerr << "Error: Failed to set callback for read event (" << err << ")" << std::endl;
            return false;
        }
    }

    return true;
}


ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra, bool luse_tf8, cl_context context) {
  accumulateDatas = read_config("./runners/accumulateDatas.txt");
  waitRecovery = read_config("./runners/waitRecovery.txt");
  collectData = read_config("./runners/collectData.txt");

  std::cerr << "accumulate data : " << accumulateDatas << std::endl;
  std::cerr << "wait recovery : " << waitRecovery << std::endl;
  std::cerr << "collect data : " << collectData << std::endl;
  fst::create_directory(LOGROOT);

  thneed = new Thneed(true, context);
  thneed->load(path);
  thneed->clexec();

  ImgSize = thneed->input_sizes[3]; 
  FEATURE_SIZE = thneed->input_sizes[0];
  TRAFFIC_SIZE = thneed->input_sizes[1];
  DESIRE_SIZE = thneed->input_sizes[2];
  clGetMemObjectInfo(thneed->output, CL_MEM_SIZE, sizeof(OUTPUT_SIZE), &OUTPUT_SIZE, NULL);
  FILE_SIZE = FEATURE_SIZE + TRAFFIC_SIZE + DESIRE_SIZE + OUTPUT_SIZE;

  std::cerr << "FEATURE_SIZE : " << FEATURE_SIZE << std::endl;
  std::cerr << "TRAFFIC_SIZE : " << TRAFFIC_SIZE << std::endl;
  std::cerr << "DESIRE_SIZE : " << DESIRE_SIZE << std::endl;
  std::cerr << "OUTPUT_SIZE : " << OUTPUT_SIZE << std::endl;
  std::cerr << "FILE_SIZE : " << FILE_SIZE << std::endl;
  if (luse_extra){
  IMGBUFFER = std::make_shared<std::vector<char>>(ImgSize * accumulateDatas * 2);
  } else{
  IMGBUFFER = std::make_shared<std::vector<char>>(ImgSize * accumulateDatas);
  }
  FILEBUFFER = std::make_shared<std::vector<char>>(FILE_SIZE * accumulateDatas);
  DATA = new CallbackData{IMGBUFFER, FILEBUFFER, ImgSize, 0, accumulateDatas};
  
  recorded = false;
  output = loutput;
  use_extra = luse_extra;
}

void ThneedModel::addRecurrent(float *state, int state_size) {
  recurrent = state;
}

void ThneedModel::addTrafficConvention(float *state, int state_size) {
  trafficConvention = state;
}

void ThneedModel::addDesire(float *state, int state_size) {
  desire = state;
}

void ThneedModel::addDrivingStyle(float *state, int state_size) {
    drivingStyle = state;
}

void ThneedModel::addNavFeatures(float *state, int state_size) {
  navFeatures = state;
}

void ThneedModel::addImage(float *image_input_buf, int buf_size) {
  input = image_input_buf;
}

void ThneedModel::addExtra(float *extra_input_buf, int buf_size) {
  extra = extra_input_buf;
}

void* ThneedModel::getInputBuf() {
  if (use_extra && thneed->input_clmem.size() > 4) return &(thneed->input_clmem[4]);
  else if (!use_extra && thneed->input_clmem.size() > 3) return &(thneed->input_clmem[3]);
  else return nullptr;
}

void* ThneedModel::getExtraBuf() {
  if (thneed->input_clmem.size() > 3) return &(thneed->input_clmem[3]);
  else return nullptr;
}

void ThneedModel::execute() {
  long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  if (!recorded) {
    thneed->record = true;
    if (use_extra) {
      float *inputs[5] = {recurrent, trafficConvention, desire, extra, input};
      thneed->copy_inputs(inputs);
    } else {
      float *inputs[4] = {recurrent, trafficConvention, desire, input};
      thneed->copy_inputs(inputs);
    }
    thneed->clexec();
    thneed->copy_output(output);
    thneed->stop();

    recorded = true;
  } else {

    if (use_extra) {

      float *inputs[5] = {recurrent, trafficConvention, desire, extra, input};
      thneed->execute(inputs, output);
      size_t current_offset;
      //std::cerr << " ms - finishStamp: " << ms-finishStamp << ", ready to save : " << readyToSave << std::endl;
      if (collectData == 1 and (ms - finishStamp) > waitRecovery * 1000 and readyToSave == 1) {
        current_offset = save_to_buffer(recurrent, 0, FEATURE_SIZE);
        current_offset = save_to_buffer(trafficConvention, current_offset, TRAFFIC_SIZE);
        current_offset = save_to_buffer(desire, current_offset, DESIRE_SIZE);
        current_offset = save_to_buffer(output, current_offset, OUTPUT_SIZE);
      	save_clmem_to_file(thneed->input_clmem[3], thneed->context, thneed->command_queue, false);
        save_clmem_to_file(thneed->input_clmem[4], thneed->context, thneed->command_queue, true);
      }

    } else {
      float *inputs[4] = {recurrent, trafficConvention, desire, input};
      thneed->execute(inputs, output);
      
    }
  }
}

