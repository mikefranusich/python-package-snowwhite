from snowwhite.mddftsolver import *
import sys
import os

if (len(sys.argv) < 2) or (sys.argv[1] == "?"):
    script = os.path.basename(__file__)
    print(script + " sz [ F|I [ d|s ]]")
    print("  sz is N or N1,N2,N3")
    print("  F  = Forward, I = Inverse")
    print("  d  = double, s = single precision")
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
    
if len(sys.argv) > 2:
    if sys.argv[2] == "I":
        k = SW_INVERSE
        
if len(sys.argv) > 3:
    if sys.argv[3] == "s":
        c_type = 'float'
        cxtype = np.csingle

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


print("CPU", funcname_cpu, libname_cpu, funcptr_cpu)
print("CUDA", funcname_cuda, libname_cuda, funcptr_cuda)
print("HIP", funcname_hip, libname_hip, funcptr_hip)


        