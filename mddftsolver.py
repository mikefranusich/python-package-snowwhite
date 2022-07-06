
from snowwhite import *
from snowwhite.swsolver import *
import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


class MddftProblem(SWProblem):
    """Define Multi-dimention DFT problem."""

    def __init__(self, ns, k=SW_FORWARD):
        """Setup problem specifics for MDDFT solver.
        
        Arguments:
        ns     -- dimensions of MDDFT
        """
        super(MddftProblem, self).__init__()
        self._ns = ns
        self._k = k
        
    def dimensions(self):
        return self._ns
        
    def direction(self):
        return self._k
        

class MddftSolver(SWSolver):
    def __init__(self, problem: MddftProblem, opts = {}):
        if not isinstance(problem, MddftProblem):
            raise TypeError("problem must be an MddftProblem")
        
        typ = 'z'
        if opts.get(SW_OPT_REALCTYPE, 0) == 'float':
            typ = 'c'
        ns = 'x'.join([str(n) for n in problem.dimensions()])
        namebase = ''
        if problem.direction() == SW_FORWARD:
            namebase = typ + 'mddft_fwd_' + ns
        else:
            namebase = typ + 'mddft_inv_' + ns
        
        if opts.get(SW_OPT_COLMAJOR, False):
            namebase = namebase + '_F'
            
        opts[SW_OPT_METADATA] = True
                    
        super(MddftSolver, self).__init__(problem, namebase, opts)

    def runDef(self, src):
        """Solve using internal Python definition."""
        
        xp = get_array_module(src)

        if self._problem.direction() == SW_FORWARD:
            FFT = xp.fft.fftn ( src )
        else:
            FFT = xp.fft.ifftn ( src ) 

        return FFT
        
    def _trace(self):
        pass

    def solve(self, src):
        """Call SPIRAL-generated function."""
        
        xp = get_array_module(src)

        nt = tuple(self._problem.dimensions())
        ordc = 'F' if self._colMajor else 'C'
        dst = xp.zeros(nt, src.dtype,  order=ordc)
        self._func(dst, src)
        if self._problem.direction() == SW_INVERSE:
            xp.divide(dst, xp.size(dst), out=dst)
        return dst

    def _writeScript(self, script_file):
        filename = self._namebase
        nameroot = self._namebase
        dims = str(self._problem.dimensions())
        filetype = '.c'
        if self._genCuda:
            filetype = '.cu'
        
        print("Load(fftx);", file = script_file)
        print("ImportAll(fftx);", file = script_file) 
        if self._genCuda:
            print("conf := LocalConfig.fftx.confGPU();", file = script_file) 
        else:
            print("conf := LocalConfig.fftx.defaultConf();", file = script_file) 
        print("t := let(ns := " + dims + ",", file = script_file) 
        print('    name := "' + nameroot + '",', file = script_file)
        # -1 is inverse for Numpy and forward (1) for Spiral
        if self._colMajor:
            print("    TFCall(TRC(TColMajor(MDDFT(ns, " + str(self._problem.direction() * -1) + "))), rec(fname := name, params := []))", file = script_file)
        else:
            print("    TFCall(TRC(MDDFT(ns, " + str(self._problem.direction() * -1) + ")), rec(fname := name, params := []))", file = script_file)
        print(");", file = script_file)        

        print("opts := conf.getOpts(t);", file = script_file)
        if self._genCuda:
            print('opts.wrapCFuncs := true;', file = script_file)
        if self._opts.get(SW_OPT_REALCTYPE) == "float":
            print('opts.TRealCtype := "float";', file = script_file)
        print("tt := opts.tagIt(t);", file = script_file)
        print("", file = script_file)
        print("c := opts.fftxGen(tt);", file = script_file)
        print('PrintTo("' + filename + filetype + '", opts.prettyPrint(c));', file = script_file)
        print("", file = script_file)
        
    def _functionMetadata(self):
        obj = dict()
        obj[SW_KEY_TRANSFORMTYPE] = SW_TRANSFORM_MDDFT
        obj[SW_KEY_DIRECTION]  = SW_STR_INVERSE if self._problem.direction() == SW_INVERSE else SW_STR_FORWARD
        obj[SW_KEY_PRECISION] = SW_STR_SINGLE if self._opts.get(SW_OPT_REALCTYPE) == "float" else SW_STR_DOUBLE
        obj[SW_KEY_DIMENSIONCOUNT] = len(self._problem.dimensions())
        obj[SW_KEY_DIMENSIONS] = self._problem.dimensions()
        names = dict()
        obj[SW_KEY_NAMES] = names
        names[SW_KEY_EXEC] = self._namebase
        names[SW_KEY_INIT] = 'init_' + self._namebase
        names[SW_KEY_DESTROY] = 'destroy_' + self._namebase
        return obj
        
