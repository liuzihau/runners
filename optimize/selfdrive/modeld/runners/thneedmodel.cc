#include "selfdrive/modeld/runners/thneedmodel.h"

#include <cassert>

//
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <vector>
#include <CL/cl.h>
#include <memory>
#include <atomic>


const int DESIRE_SIZE_A = 1 * 100 * 8;
const int TRAFFIC_SIZE_A = 1 * 2;
const int FEATURE_SIZE_A = 1 * 99 * 128;
const int OUTPUT_SIZE_A = 1 * 6108;


const std::string LOGROOT = "/data/openpilot_log";
namespace fst = std::filesystem;

std::atomic<bool> write_extra_complete = true;
std::atomic<bool> write_input_complete = true;


void save_array_to_file(std::string file_path, const float* array_obj, const int array_obj_size) {
  // Open the output file stream
  std::ofstream output_file(file_path, std::ios::binary);

  // Write the array's contents to the output file
  output_file.write(reinterpret_cast<const char*>(array_obj), array_obj_size * sizeof(float));

  // Close the output file stream
  output_file.close();
}


struct CallbackData {
    std::shared_ptr<std::vector<char>> buffer;
    std::string file_path;
    bool extra;
};

void CL_CALLBACK write_complete_callback(cl_event event, cl_int status, void *user_data) {
    cl_int err;
    if (status != CL_SUCCESS) {
        std::cerr << "Error: Failed to complete writing data to file (" << status << ")" << std::endl;
        return;
    }

    auto data = static_cast<CallbackData*>(user_data);
    auto buffer = data->buffer;
    auto extra = data->extra;
    const std::string& file_path = data->file_path;

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

    if (extra){
    write_extra_complete = true;
    } else{
    write_input_complete = true;
    }
}

bool save_clmem_to_file(const cl_mem cl_mem_obj, size_t file_size, const std::string& file_path, cl_context context, cl_command_queue command_queue, bool extra) {
    cl_int err;

    // Allocate a large enough memory buffer
    auto buffer = std::make_shared<std::vector<char>>(file_size);

    // Read cl_mem_obj into buffer
    cl_event read_event;
    err = clEnqueueReadBuffer(command_queue, cl_mem_obj, CL_FALSE, 0, file_size, buffer->data(), 0, nullptr, &read_event);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to read cl_mem_obj (" << err << ")" << std::endl;
        return false;
    }

    // Set callback function to write to file after read operation is complete
    auto data = new CallbackData{buffer, file_path, extra};
    err = clSetEventCallback(read_event, CL_COMPLETE, &write_complete_callback, data);
    if (err != CL_SUCCESS) {
        std::cerr << "Error: Failed to set callback for read event (" << err << ")" << std::endl;
        delete data;
        return false;
    }

    return true;
}

bool save_all(Thneed* thneed, float** inputs, float* desire, float* trafficConvention, float* recurrent, float* output, const std::string& folder, bool use_extra){
      fst::create_directory(folder);
      write_extra_complete = false;
      write_input_complete = false;
      if (use_extra) {
        save_array_to_file(folder + "/" + "desire.bin",desire, DESIRE_SIZE_A);
        save_array_to_file(folder + "/" + "traffic_convention.bin", trafficConvention, TRAFFIC_SIZE_A);
        save_array_to_file(folder + "/" + "features_buffer.bin", recurrent, FEATURE_SIZE_A);
        save_clmem_to_file(thneed->input_clmem[3], thneed->input_sizes[3], folder + "/" + "big_input_imgs.bin", thneed->context, thneed->command_queue, true);
        save_clmem_to_file(thneed->input_clmem[4], thneed->input_sizes[4], folder + "/" + "input_imgs.bin", thneed->context, thneed->command_queue, false);
        save_array_to_file(folder + "/" + "output.bin", output, OUTPUT_SIZE_A);
    } else {
        save_array_to_file(folder + "/" + "desire.bin",desire, DESIRE_SIZE_A);
        save_array_to_file(folder + "/" + "traffic_convention.bin", trafficConvention, TRAFFIC_SIZE_A);
        save_array_to_file(folder + "/" + "features_buffer.bin", recurrent, FEATURE_SIZE_A);
        save_clmem_to_file(thneed->input_clmem[3], thneed->input_sizes[3], folder + "/" + "input_imgs.bin", thneed->context, thneed->command_queue, true);
        save_array_to_file(folder + "/" + "output.bin", output, OUTPUT_SIZE_A);
      }
      return true;
}



ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra, bool luse_tf8, cl_context context) {
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
  //    
  long ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  const std::string SESSION = std::to_string(ms);
  const std::string folder = LOGROOT + "/" + SESSION;
  //if (ms - check_time > 2000) {
  //}
  //	
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
      if (write_extra_complete == true and write_input_complete == true){
	save_all(thneed, inputs, desire, trafficConvention, recurrent, output, folder, use_extra);
	}
    } else {
      float *inputs[4] = {recurrent, trafficConvention, desire, input};
      thneed->execute(inputs, output);
      if (write_input_complete == true){
	      save_all(thneed, inputs, desire, trafficConvention, recurrent, output, folder, use_extra);
      }
    }
  }
}

