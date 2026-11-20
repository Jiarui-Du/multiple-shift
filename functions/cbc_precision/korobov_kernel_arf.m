function omega=korobov_kernel_arf(alpha)
% Native high-precision Korobov kernel handle for fastrank1pt_arf.
% The CBC wrapper evaluates rational grid points in the native runtime.
% Direct omega(x) calls return double values after native kernel evaluation.
validateattributes(alpha,{'double'},{'scalar','real','finite','>',0.5});
arf_kernel_descriptor=struct('tag','arf-korobov-v1','alpha',alpha);
omega=@evaluate;
    function values=evaluate(x)
        if nargin==0,values=arf_kernel_descriptor;return;end
        validateattributes(x,{'double'},{'real','finite'});
        runtime=arf_cbc_runtime();old=getenv('PATH');
        cleanup=onCleanup(@()setenv('PATH',old));
        setenv('PATH',[runtime.directory pathsep old]);
        values=fastrank1pt_arf_mex('kernel',runtime.dll,arf_kernel_descriptor.alpha,x,113,8);
    end
end
