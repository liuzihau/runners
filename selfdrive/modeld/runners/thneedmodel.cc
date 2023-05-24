#include "selfdrive/modeld/runners/thneedmodel.h"

#include <cassert>

#include <iostream>
#include <fstream>
#include <random>
#include <string>

#include <filesystem>

int accumulateDatas = 100; // 100 frames
int waitRecovery = 10; // 10s
int collectData = 1; // 1:true, 0:false
int readyToSave = 1; // 1:true, 0:false
long finishStamp = 0; 

const std::string LOGROOT = "/home/openpilot_log";
namespace fst = std::filesystem;

std::string SESSION;
std::string FOLDER;

const size_t FileSize = 4 * 1 * 12 * 128 * 256;
std::shared_ptr<std::vector<char>> BUFFER = std::make_shared<std::vector<char>>(FileSize * accumulateDatas); // Create buffer [accumulateDatas] * file size larger

struct CallbackData {
    std::shared_ptr<std::vector<char>> buffer;
    size_t file_size;
    size_t files_written;
    int max_files;

    CallbackData(std::shared_ptr<std::vector<char>> buf, size_t size, size_t written, int max)
    : buffer(buf), file_size(size), files_written(written), max_files(max)
    {}
};

CallbackData* IMG_DATA = new CallbackData{BUFFER, FileSize, 0, accumulateDatas};

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

void save_array_to_file(std::string file_path, const float* array_obj, const int array_obj_size) {
  std::ofstream output_file(file_path, std::ios::binary);
  output_file.write(reinterpret_cast<const char*>(array_obj), array_obj_size * sizeof(float));
  output_file.close();
}

bool copy_clmem_to_buffer(const cl_mem cl_mem_obj, size_t file_size,cl_command_queue command_queue) {
    cl_int err;
    auto buffer = std::make_shared<std::vector<char>>(file_size);
    err = clEnqueueReadBuffer(command_queue, cl_mem_obj, CL_TRUE, 0, file_size, buffer->data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to read cl_mem_obj (" << err << ")" << std::endl;
        return false;
    }

    return true;
}


void CL_CALLBACK write_complete_callback(cl_event event, cl_int status, void *user_data) {
    cl_int err;
    
    readyToSave = 0;
    fst::create_directory(FOLDER);

    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to complete writing data to file (" << status << ")" << std::endl;
        return;
    }
    auto data = static_cast<CallbackData*>(user_data);
    auto buffer = data->buffer;
    const std::string& file_path = FOLDER + "/img_inputs.bin";

    std::ofstream output_file(file_path, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error: Failed to open file \"" << file_path << "\"" << std::endl;
        return;
    }

    output_file.write(buffer->data(), buffer->size());
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


bool save_clmem_to_file(const cl_mem cl_mem_obj, cl_context context, cl_command_queue command_queue) {
    cl_int err;

    // Write to the next section of the buffer
    size_t offset = IMG_DATA->files_written * FileSize;

    // Read cl_mem_obj into buffer
    cl_event read_event;
    //err = clEnqueueReadBuffer(command_queue, cl_mem_obj, CL_FALSE, offset, FileSize, IMG_DATA->buffer->data() + offset, 0, nullptr, &read_event);
    err = clEnqueueReadBuffer(command_queue, cl_mem_obj, CL_FALSE, 0, FileSize, reinterpret_cast<void*>(IMG_DATA->buffer->data() + offset), 0, nullptr, &read_event);

    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to read cl_mem_obj (" << err << ")" << std::endl;
        return false;
    }

    IMG_DATA->files_written++;

    // Set callback function to write to file after buffer is full
    if (IMG_DATA->files_written >= IMG_DATA->max_files and readyToSave == 1) {
        err = clSetEventCallback(read_event, CL_COMPLETE, &write_complete_callback, IMG_DATA);
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
  SESSION = std::to_string(ms);
  FOLDER = LOGROOT + "/" + SESSION;

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

      std::cerr << " ms - finishStamp: " << ms-finishStamp << ", ready to save : " << readyToSave << std::endl;
      if (collectData == 1 and (ms - finishStamp) > waitRecovery * 1000 and readyToSave == 1) {
        save_clmem_to_file(thneed->input_clmem[3], thneed->context, thneed->command_queue);
      }

    } else {
      float *inputs[4] = {recurrent, trafficConvention, desire, input};
      thneed->execute(inputs, output);
      
    }
  }
}

