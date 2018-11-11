#include "engine.h"
#include "lz77.h"
#include <stdio.h>
#include <sys/stat.h>

inline static void PrintMatch(const lz77_match_t* match) {
  fprintf(stderr, "{offset: %d, length: %d, next: %02x}\n", match->offset,
          match->length, match->next & 0xff);
}

static inline int FindDevice(opencl_engine_t* engine) {
  cl_platform_id platforms[2];
  int err = clGetPlatformIDs(2, &platforms, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to get platforms!\n");
    return -1;
  }

  // Connect to a compute device
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
  engine->commands =
      clCreateCommandQueue(engine->context, engine->device_id, 0, &err);
  if (!engine->commands) {
    printf("Error: Failed to create a command commands!\n");
    return -2;
  }
  return 0;
}

opencl_engine_t CreateEngine() {
  opencl_engine_t engine;
  FindDevice(&engine);
  CreateContext(&engine);
  return engine;
}

void DestroyEngine(opencl_engine_t* engine) {
  clReleaseCommandQueue(engine->commands);
  clReleaseContext(engine->context);
}

void DestroyCodec(opencl_codec_t* codec) {
  clReleaseProgram(codec->program);
  clReleaseKernel(codec->kernel);
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

static int EncodeLZ77(struct opencl_codec* codec, const char* in, unsigned in_len, char* out, unsigned out_len) {
    // Create the input and output arrays in device memory for our calculation
    cl_mem input = clCreateBuffer(codec->engine->context,  CL_MEM_READ_ONLY,  in_len, NULL, NULL);
    cl_mem output = clCreateBuffer(codec->engine->context, CL_MEM_WRITE_ONLY, sizeof(lz77_match_t)*in_len, NULL, NULL);
    if (!input || !output)
    {
        printf("Error: Failed to allocate device memory!\n");
        return -1;
    }
    // Write our data set into the input array in device memory
    int err = clEnqueueWriteBuffer(codec->engine->commands, input, CL_TRUE, 0, in_len, in, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to write to source array!\n");
        return -2;
    }

    // Set the arguments to our compute kernel
    err = 0;
    err  = clSetKernelArg(codec->kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(codec->kernel, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(codec->kernel, 2, sizeof(unsigned int), &in_len);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return -3;
    }

    size_t local = 0;
    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(codec->kernel, codec->engine->device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return -4;
    }

    // Execute the kernel over the entire range of our 1d input data set
    // using the maximum number of work group items for this device
    size_t global = in_len;
    if(local > global)
        local = global;
    printf("global size: %ld, local size: %ld\n", global, local);
    err = clEnqueueNDRangeKernel(codec->engine->commands, codec->kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err)
    {
        printf("Error: Failed to execute kernel! Error: %d\n", err);
        return -5;
    }

    // Wait for the command commands to get serviced before reading back results
    clFinish(codec->engine->commands);

    // Read back the results from the device to verify the output
    /* TODO XXX readback size and out_len need to be reconciled */
    int readback_size = in_len * sizeof(lz77_match_t);
    err = clEnqueueReadBuffer(codec->engine->commands, output, CL_TRUE, 0, readback_size, out, 0, NULL, NULL );
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output array! %d\n", err);
        return -6;
    }
    clReleaseMemObject(input);
    clReleaseMemObject(output);

    int match_count = readback_size / sizeof(lz77_match_t);
    for(int i = 0; i < match_count; i++) {
        lz77_match_t match = ((lz77_match_t*) out)[i];
        PrintMatch(&match);
    }

  return 0;
}

static int DecodeLZ77(struct opencl_codec* codec, const char* in, unsigned in_len, char* out, unsigned out_len) {
  return 0;
}

opencl_codec_t GetCodec(opencl_engine_t* engine, codec_name_t name) {
  opencl_codec_t codec = {.state = INVALID, .Encode = NULL, .Decode = NULL, .engine = engine};
  fprintf(stderr, "Loading Codec %d\n", name);
  switch (name) {
    case LZ77: {
      int err = BuildProgram(engine, &codec, "lz77.cl", "Encode");
      if (err != 0) {
        fprintf(stderr, "Failed Building Codec\n");
        return codec;
      }
      codec.Encode = EncodeLZ77;
      codec.Decode = DecodeLZ77;
      break;
    }
    default:
      fprintf(stderr, "Unknown codec %d\n", name);
  }
  codec.state = READY;
  return codec;
}
