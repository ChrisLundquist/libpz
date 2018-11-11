__kernel void Encode(__global const char *in,
                     __global char *out,
                              const unsigned count) {
  int i = get_global_id(0);
  if(i > count)
    return;

  out[i] = in[i];
}
