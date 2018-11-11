#include <stdio.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "engine.h"

static inline int FindDevice(opencl_engine_t* engine) {
  cl_platform_id platforms[2];
  int err = clGetPlatformIDs(2, &platforms, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to get platforms!\n");
    return -1;
  }

  // Connect to a compute device
  //
  int gpu = 1;
  err = clGetDeviceIDs(platforms[0],
                       gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1,
                       &engine->device_id, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to create a device group!\n");
    return -2;
  }
  return 0;
}

static inline int CreateContext(opencl_engine_t* engine) {
  int err = 0;
  engine->context = clCreateContext(0, 1, &engine->device_id, NULL, NULL, &err);
  if (!engine->context) {
    printf("Error: Failed to create a compute context!\n");
    return -1;
  }

  // Create a command commands
  //
  engine->commands =
      clCreateCommandQueue(engine->context, engine->device_id, 0, &err);
  if (!engine->commands) {
    printf("Error: Failed to create a command commands!\n");
    return -2;
  }
  return 0;
}

static char* ReadProgramFile(const char* filepath) {
  struct stat sb;
  if (stat(filepath, &sb) != 0) {
    fprintf(stderr, "Failed opening %s. Skipping\n", filepath);
    return NULL;
  }

  char* text = malloc(sb.st_size);
  if (text == NULL) {
    fprintf(stderr, "Failed allocating memory %s. Skipping\n", filepath);
    return NULL;
  }
  FILE* file = fopen(filepath, "r");
  fread(text, 1, sb.st_size, file);
  fclose(file);
  return text;
}

/* FIXME this interface isn't quite right. we *want* to return a opencl_codec_t,
 * but have to do erros too */
static int BuildProgram(opencl_engine_t* engine,
                        opencl_codec_t* codec,
                        const char* path,
                        const char* kernel_name) {
  char* source = ReadProgramFile(path);
  if (source == NULL)
    return -1;

  int err = 0;
  codec->program =
      clCreateProgramWithSource(engine->context, 1, &source, NULL, &err);
  free(source);
  if (!codec->program) {
    fprintf(stderr, "Error: Failed to create compute program!\n");
    return -2;
  }

  // Build the program executable
  err = clBuildProgram(codec->program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[2048];

    fprintf(stderr, "Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(codec->program, engine->device_id,
                          CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    fprintf(stderr, "%s\n", buffer);
    return -3;
  }

  // Create the compute kernel in the program we wish to run
  codec->kernel = clCreateKernel(codec->program, kernel_name, &err);
  if (!codec->kernel || err != CL_SUCCESS) {
    fprintf(stderr, "Error: Failed to create compute kernel!\n");
    return -4;
  }
  return 0;
}
