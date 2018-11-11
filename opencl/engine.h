#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "codec.h"

typedef struct {
    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
} opencl_engine_t;

typedef struct {
    codec_name_t name;
    codec_state_t state;
    cl_program program; // compute program
    cl_kernel kernel;   // compute kernel
    int (*Encode)(const char* plain, unsigned plain_len, const char* out, unsigned out_len);
    int (*Decode)(const char* encoded, unsigned encoded_len, const char* out, unsigned out_len);
} opencl_codec_t;

opencl_engine_t CreateEngine();
opencl_codec_t GetCodec(opencl_engine_t *engine, codec_name_t name);

void DestroyEngine(opencl_engine_t *engine);

void DestroyCodec(opencl_codec_t *codec);
