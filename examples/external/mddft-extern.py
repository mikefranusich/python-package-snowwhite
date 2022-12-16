from snowwhite.mddftsolver import *
import numpy as np
import sys
import os
import ctypes

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    script = os.path.basename(__file__)
    print(script + " sz [ F|I [ d|s ]]")
    print("  sz is N or N1,N2,N3")
    sys.exit()

# direction, SW_FORWARD or SW_INVERSE
k = SW_FORWARD
# base C type, 'float' or 'double'
c_type = 'double'
cxtype = np.cdouble

nnn = sys.argv[1].split(',')

n1 = int(nnn[0])
n2 = (lambda:n1, lambda:int(nnn[1]))[len(nnn) > 1]()
n3 = (lambda:n2, lambda:int(nnn[2]))[len(nnn) > 2]()

dims = [n1,n2,n3]
dimsTuple = tuple(dims)

problem = MddftProblem(dims, k)

opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : SW_CPU }
try:
    s = MddftSolver(problem, opts)
    solver_cpu = s
    funcname_cpu = s._namebase
    libname_cpu = os.path.join(s._libsDir, 'lib' + s._namebase + SW_SHLIB_EXT)
    funcptr_cpu = s._MainFunc
except:
    print("unable to build CPU solver")
    solver_cpu = None
    funcname_cpu = ""
    libname_cpu  = ""
    funcptr_cpu = None

    
opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : SW_CUDA }
try:
    s = MddftSolver(problem, opts)
    solver_cuda = s
    funcname_cuda = s._namebase
    libname_cuda = os.path.join(s._libsDir, 'lib' + s._namebase + SW_SHLIB_EXT)
    funcptr_cuda = s._MainFunc
except:
    print("unable to build CUDA solver")
    solver_cuda = None
    funcname_cuda = ""
    libname_cuda = ""
    funcptr_cuda = None

    
opts = { SW_OPT_REALCTYPE : c_type, SW_OPT_PLATFORM : SW_HIP }
try:
    s = MddftSolver(problem, opts)
    solver_hip = s
    funcname_hip = s._namebase
    libname_hip = os.path.join(s._libsDir, 'lib' + s._namebase + SW_SHLIB_EXT)
    funcptr_hip = s._MainFunc
except:
    print("unable to build HIP solver")
    solver_hip  = None
    funcname_hip = ""
    libname_hip = ""
    funcptr_hip = None

dst = np.zeros(dimsTuple, cxtype)
src = np.ones(dimsTuple, cxtype)
for  x in range (np.size(src)):
    vr = np.random.random()
    vi = np.random.random()
    src.itemset(x,vr + vi * 1j)

libsdir = os.getcwd()
libpath = os.path.join(libsdir, 'libtestfuncs' + SW_SHLIB_EXT)
sharedlib = ctypes.CDLL(libpath)
print('Test library:', sharedlib)
print('Calling external function callfuncsbyname()\n');

#void callfuncsbyname(
#    int bytes_in,
#    int bytes_out,
#    double* ptr_in,
#    double* ptr_out,
#    char* library_cpu,
#    char* library_cuda,
#    char* library_hip,
#    char* funcname_cpu,
#    char* funcname_cuda,
#    char* funcname_hip)


callfuncsbyname = getattr(sharedlib, "callfuncsbyname")

callfuncsbyname(src.size * src.itemsize, 
                dst.size * dst.itemsize, 
                src.ctypes.data_as(ctypes.c_void_p),
                dst.ctypes.data_as(ctypes.c_void_p),
                libname_cpu.encode(),
                libname_cuda.encode(),
                libname_hip.encode(),
                funcname_cpu.encode(),
                funcname_cuda.encode(),
                funcname_hip.encode())

print('\nCalling external function callfuncs()\n');

#void callfuncs(
#    int bytes_in,
#    int bytes_out,
#    double* ptr_in,
#    double* ptr_out,
#    func2dbls func_cpu,
#    func2dbls func_cuda,
#    func2dbls func_hip)

callfuncs = getattr(sharedlib, "callfuncs")

callfuncs(src.size * src.itemsize, 
          dst.size * dst.itemsize, 
          src.ctypes.data_as(ctypes.c_void_p),
          dst.ctypes.data_as(ctypes.c_void_p),
          funcptr_cpu,
          funcptr_cuda,
          funcptr_hip)

dstP = solver_cpu.runDef(src)

diff = np.max ( np.absolute ( dst - dstP ) )
print ('Diff between NumPy/Spiral transforms = ' + str(diff) )




        