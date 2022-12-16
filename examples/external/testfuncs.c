#include <stdlib.h>
#include <stdio.h>


void callfuncsbyname(
    int bytes_in,
    int bytes_out,
    double* ptr_in,
    double* ptr_out,
    char* library_cpu,
    char* library_cuda,
    char* library_hip,
    char* funcname_cpu,
    char* funcname_cuda,
    char* funcname_hip)
{
    // library function call args: (ptr_out, ptr_in)
    
    printf("bytes_in: %d, bytes_out: %d\n", bytes_in, bytes_out);
    printf("ptr_in: %p, ptr_out: %p\n", ptr_in, ptr_out);
    printf("CPU lib/func: %s, %s\n", library_cpu, funcname_cpu);
    printf("CUDA lib/func: %s, %s\n", library_cuda, funcname_cuda);
    printf("HIP lib/func: %s, %s\n", library_hip, funcname_hip);
}  

typedef void (*func2dbls)(double *, double *);

void callfuncs(
    int bytes_in,
    int bytes_out,
    double* ptr_in,
    double* ptr_out,
    func2dbls func_cpu,
    func2dbls func_cuda,
    func2dbls func_hip)
{
    // library function call args: (ptr_out, ptr_in)
  
    // call the CPU version, in/out data is on CPU
    if (func_cpu != 0) {
        func_cpu(ptr_out, ptr_in);
    }
}
    

    