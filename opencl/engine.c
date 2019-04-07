#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "engine.h"
#include "lz77.h"

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
  // XXX TODO check OOQ is supported
  const cl_queue_properties ooq[] = {CL_QUEUE_PROPERTIES,
                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
  // const cl_queue_properties iiq[] = {CL_QUEUE_PROPERTIES, 0};
  engine->commands = clCreateCommandQueueWithProperties(
      engine->context, engine->device_id, &ooq, &err);

  if (!engine->commands || err < 0) { /* doesn't support OOQ */
    printf("Falling back to In-Order-Queue\n");
    engine->commands = clCreateCommandQueueWithProperties(
        engine->context, engine->device_id, 0, &err);
  }

  if (!engine->commands) {
    printf("Error: Failed to create a command commands! Err: %d\n", err);
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
  char* source = (char*)ReadProgramFile(path);
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
  // -cl-std=CL1.2
  err = clBuildProgram(codec->program, 0, NULL, "-Werror", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t len;
    char buffer[4096];

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

inline void PrintMatch(const lz77_match_t* match) {
  fprintf(stderr, "{offset: %u, length: %u, next: %02x}\n", match->offset,
          match->length, match->next & 0xff);
}

inline void PrintMatches(const lz77_match_t* matches, unsigned match_len) {
  for (unsigned i = 0; i < match_len; ++i) {
    fprintf(stderr, "index: %u ", i);
    PrintMatch(&matches[i]);
  }
}

// In OpenCL we will calculate a number of redundant matches.
// This happens when at index I we get a match len of 7 then at I+1 we get 6
// at I + 2 we get 5, etc.
// This finds the number of matches if we skipped by match length
static int DedupeMatches(lz77_match_t* matches, unsigned match_count) {
  /* the first match will always be a literal so we can skip it */
  unsigned count = 1;
  for (unsigned index = 1; index < match_count; ++count) {
    // fprintf(stderr, "choosing match %d ", index);
    // PrintMatch(&matches[index]);
    matches[count] = matches[index];
    index += matches[index].length + 1;
  }
  // printf("Only need %u matches (%d bytes) %g%\n", count, count *
  // sizeof(lz77_match_t), 100 * (float)count / match_count);
  return count;
}

static int EncodeLZ77(struct opencl_codec* codec,
                      const char* in,
                      unsigned in_len,
                      char* out,
                      unsigned out_len) {
  if (in_len == 0 || out_len == 0) {
    printf("Called Encode with 0 len in or out!\n");
    return 0;
  }

  // Create the input and output arrays in device memory for our calculation
  cl_mem input = clCreateBuffer(codec->engine->context, CL_MEM_READ_ONLY,
                                in_len, NULL, NULL);
  cl_mem output = clCreateBuffer(codec->engine->context, CL_MEM_WRITE_ONLY,
                                 sizeof(lz77_match_t) * in_len, NULL, NULL);
  if (!input || !output) {
    printf("Error: Failed to allocate device memory!\n");
    return -1;
  }
  // Write our data set into the input array in device memory
  int err = clEnqueueWriteBuffer(codec->engine->commands, input, CL_TRUE, 0,
                                 in_len, in, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to write to source array!\n");
    return -2;
  }

  // Set the arguments to our compute kernel
  err = clSetKernelArg(codec->kernel, 0, sizeof(cl_mem), &input);
  err |= clSetKernelArg(codec->kernel, 1, sizeof(cl_mem), &output);
  err |= clSetKernelArg(codec->kernel, 2, sizeof(unsigned int), &in_len);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    return -3;
  }

  size_t local = 0;
  // Get the maximum work group size for executing the kernel on the device
  err = clGetKernelWorkGroupInfo(codec->kernel, codec->engine->device_id,
                                 CL_KERNEL_WORK_GROUP_SIZE, sizeof(local),
                                 &local, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    return -4;
  }

  // Tweaks for edge cases
  // global has to be a multiple of local, so make sure it lines up.
  // we must protected aginst extra opencl device threads in the kernel)
  const int worker_window = 32;
  size_t global = in_len / worker_window;
  if (global % local != 0) {
    global = global - (global % local) + local;
    // global /= worker_window; // how many bytes each worker looks at
  }

  // for very small groups we need to clamp this
  if (local > global)
    local = global;
  const size_t step_size = local;
  printf("global size: %ld, local size: %ld, step size: %ld\n", global, local,
         step_size);
  // Execute the kernel over the entire range of our 1d input data set
  // using the maximum number of work group items for this device
  size_t loops = 0;
  for (size_t i = 0; i < global; i += step_size, ++loops) {
    /* TODO readback chunks of results after a kernel finishes */
    // printf(
    //    "enqueue kernel global: %ld, step_size: %ld, local: %ld, offset:
    //    %ld\n", global, step_size, local, i);
    err = clEnqueueNDRangeKernel(codec->engine->commands, codec->kernel, 1, &i,
                                 &step_size, &local, 0, NULL, NULL);
    if (err) {
      printf("Error: Failed to execute kernel! Error: %d\n", err);
      return -5;
    }
    if (loops % 8 == 0) {
      clFlush(codec->engine->commands);
    }
  }

  printf("Waiting for commands to finish\n");
  // Wait for the command commands to get serviced before reading back results
  clFinish(codec->engine->commands); /* In Order Queue */
  // clEnqueueBarrier(codec->engine->commands); /*XXX Out of Order */
  printf("Reading Results\n");
  const unsigned readback_size = in_len * sizeof(lz77_match_t);
  // const unsigned readback_size = 196 * sizeof(lz77_match_t); // XXX
  lz77_match_t* tmp_matches = malloc(readback_size);
  if (tmp_matches == NULL) {
    printf("Error: allocating temp buffer!\n");
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    return -6;
  }

  // Read back the results from the device to verify the output
  /* TODO zero copy and remove tmp */
  err = clEnqueueReadBuffer(codec->engine->commands, output, CL_TRUE, 0,
                            readback_size, tmp_matches, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    printf("Error: Failed to read output array! %d\n", err);
    return -7;
  }

  // PrintMatches(tmp_matches, readback_size / sizeof(lz77_match_t));

  unsigned deduped_count =
      DedupeMatches(tmp_matches, readback_size / sizeof(lz77_match_t));
  const unsigned deduped_size = deduped_count * sizeof(lz77_match_t);
  if (deduped_size > out_len) {
    printf("Error: output buffer doesn't have enough room need %d, have %d!\n",
           deduped_size, out_len);
    free(tmp_matches);
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    return -7;
  }

  memcpy(out, tmp_matches, deduped_size);

  printf("Releasing memory\n");
  free(tmp_matches);
  clReleaseMemObject(input);
  clReleaseMemObject(output);

  return deduped_size;
}

/* FIXME copy paste form reference */
static int DecodeLZ77(struct opencl_codec* codec,
                      const char* in,
                      unsigned insize,
                      char* out,
                      unsigned outsize) {
  lz77_match_t* matches = (lz77_match_t*)in;
  int match_size = insize / sizeof(lz77_match_t);
  register char* outpos = out;

  for (int i = 0; i < match_size; i++) {
    register lz77_match_t m = matches[i];
    if (((outpos - out) + m.length) > (outsize)) {
      fprintf(stderr, "not enough room in output buffer\n");
      break;
    }
    register char* seek = outpos - m.offset;
    for (register unsigned j = 0; j < m.length; j++) {
      *outpos = *seek;
      outpos++;
      seek++;
    }
    *outpos = m.next;
    outpos++;
  }

  return outpos - out;
}

opencl_codec_t GetCodec(opencl_engine_t* engine, codec_name_t name) {
  opencl_codec_t codec = {
      .state = INVALID, .Encode = NULL, .Decode = NULL, .engine = engine};
  fprintf(stderr, "Loading Codec %d\n", name);
  switch (name) {
    case LZ77: {
      int err = BuildProgram(engine, &codec, "lz77-batch.cl", "Encode");
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
