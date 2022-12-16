#include <stdlib.h>
#include <stdio.h>


void callfuncsbyname(
    int bytes_in,
    int bytes_out,
    void* ptr_in,
    void* ptr_out,
    char* library_cpu,
    char* library_cuda,
    char* library_hip,
    char* funcname_cpu,
    char* funcname_cuda,
    char* funcname_hip)
{
    // library function call args: (ptr_out, ptr_in)
    
    printf("bytes_in: %d, bytes_out: %d\n", bytes_in, bytes_out);
    printf("ptr_in: %X, ptr_out: %X\n", ptr_in, ptr_out);
    printf("CPU lib/func: %s, %s\n", library_cpu, funcname_cpu);
    printf("CUDA lib/func: %s, %s\n", library_cuda, funcname_cuda);
    printf("HIP lib/func: %s, %s\n", library_hip, funcname_hip);
}  
    