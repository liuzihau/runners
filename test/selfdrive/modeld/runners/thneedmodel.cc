#include "selfdrive/modeld/runners/thneedmodel.h"

#include <cassert>

#include <iostream>
#include <fstream>
#include <random>
#include <string>

#include <filesystem>


const std::string LOGROOT = "/data/openpilot_log";
namespace fst = std::filesystem;


int config = 0;
const int FileSize = 1 * 12 * 128 * 256;
float dummy1[FileSize];
float dummy2[FileSize];

int read_config() {
    //std::filesystem::path current_path = std::filesystem::current_path();
    //std::cerr << current_path;
    std::ifstream ifs;
    char buffer[256] = {0};
    ifs.open("./runners/config.txt");
    if (!ifs.is_open()) {
	std::cerr << "Failed to open file.\n";
        return 1; // EXIT_FAILURE
    }
    ifs.read(buffer, sizeof(buffer));
    std::cerr << buffer << std::endl;
    int ic = buffer[0] - '0';
    ifs.close();
    return ic;
}

auto fill_dummy(float* dummy){
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 25);
    for(int i = 0; i < FileSize; i++){
        dummy[i] = dist(e2);
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
    std::cerr << "copy buffer ok" << std::endl;

    return true;
}




ThneedModel::ThneedModel(const char *path, float *loutput, size_t loutput_size, int runtime, bool luse_extra, bool luse_tf8, cl_context context) {
  config = read_config();
  fst::create_directory(LOGROOT);
  if (config == 0){
      fill_dummy(dummy1);
      fill_dummy(dummy2);
  }

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
  const std::string SESSION = std::to_string(ms);

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

      if (config == 0){
        const std::string folder = LOGROOT + "/" + "dummy_file_" + SESSION;
        fst::create_directory(folder);
        save_array_to_file(folder + "/" + "dummy1.bin", dummy1, FileSize);
        save_array_to_file(folder + "/" + "dummy2.bin", dummy2, FileSize);
      }
      if (config == 1){
        const std::string folder = LOGROOT + "/" + "no_action_" + SESSION;
        fst::create_directory(folder);

      }
      if (config == 2){
        const std::string folder = LOGROOT + "/" + "move_only_" + SESSION;
        fst::create_directory(folder);
        copy_clmem_to_buffer(thneed->input_clmem[3], thneed->input_sizes[3], thneed->command_queue);
        copy_clmem_to_buffer(thneed->input_clmem[4], thneed->input_sizes[4], thneed->command_queue);
      }

    } else {
      float *inputs[4] = {recurrent, trafficConvention, desire, input};
      thneed->execute(inputs, output);
      
      if (config == 0){
        const std::string folder = LOGROOT + "/" + "dummy_file_" + SESSION;
        fst::create_directory(folder);
        save_array_to_file(folder + "/" + "dummy1.bin", dummy1, FileSize);
        save_array_to_file(folder + "/" + "dummy2.bin", dummy2, FileSize);
      }
      if (config == 1){
        const std::string folder = LOGROOT + "/" + "no_action_" + SESSION;
        fst::create_directory(folder);

      }
      if (config == 2){
        const std::string folder = LOGROOT + "/" + "move_only_" + SESSION;
        fst::create_directory(folder);
        copy_clmem_to_buffer(thneed->input_clmem[3], thneed->input_sizes[3], thneed->command_queue);
        copy_clmem_to_buffer(thneed->input_clmem[4], thneed->input_sizes[4], thneed->command_queue);
      }
    }
  }
}

