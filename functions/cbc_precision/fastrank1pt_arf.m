function [z,e2,info,prepared]=fastrank1pt_arf(n,s_max,omega,gamma,beta,options)
% Same five core arguments as fastrank1pt; point arithmetic in a native MEX.
% e2 is converted to double only AFTER high-precision P evaluation.
% info.P preserves decimal strings. Use korobov_kernel_arf for native kernels.
arguments
    n (1,1) double {mustBeInteger,mustBePositive}
    s_max (1,1) double {mustBeInteger,mustBePositive}
    omega (1,1) function_handle
    gamma double
    beta double
    options.Bits (1,1) double {mustBeInteger,mustBeGreaterThanOrEqual(options.Bits,32),mustBeLessThanOrEqual(options.Bits,512)} = 113
    options.Threads (1,1) double {mustBeInteger,mustBeGreaterThanOrEqual(options.Threads,1),mustBeLessThanOrEqual(options.Threads,64)} = 8
    options.Audit (1,1) logical = true
    options.Prepared = []
    options.KernelCache (1,1) string = ""
end
assert(n>=3 && isprime(n) && n^2<flintmax,'arf_cbc:Grid','Require an odd prime with N^2<flintmax.');
assert(numel(gamma)==s_max && numel(beta)==s_max && isreal(gamma) && isreal(beta) && ...
    all(isfinite(gamma(:)) & gamma(:)>0) && all(isfinite(beta(:)) & beta(:)>0), ...
    'arf_cbc:Weights','Positive finite product weights of length s_max are required.');
gamma=gamma(:);beta=beta(:);descriptor=kernel_descriptor(omega);
signature=struct('N',n,'dimension',s_max,'gamma',gamma,'beta',beta, ...
    'bits',options.Bits,'threads',options.Threads,'descriptor',descriptor,'omega',omega);
runtime=arf_cbc_runtime();old_path=getenv('PATH');
restore=onCleanup(@()setenv('PATH',old_path));setenv('PATH',[runtime.directory pathsep old_path]);
started=tic;reused=~isempty(options.Prepared);
if reused
    prepared=options.Prepared;
    assert(isa(prepared,'arf_cbc_plan') && isscalar(prepared) && isvalid(prepared) && ...
        ~isempty(prepared.Id),'arf_cbc:Prepared','Prepared must be a live arf_cbc_plan.');
    expected=prepared.Signature;
    fields={'N','dimension','gamma','beta','bits','threads','descriptor'};
    for j=1:numel(fields)
        name=fields{j};assert(isequal(signature.(name),expected.(name)), ...
            'arf_cbc:Prepared','Prepared plan differs in %s.',name);
    end
    if isempty(fieldnames(descriptor))
        assert(isequal(omega,expected.omega),'arf_cbc:Prepared','Prepared kernel callback differs.');
    end
    preparation_seconds=0;source=prepared.Preparation.source;
    if strlength(options.KernelCache)>0
        assert(isfield(source,'filename') && strcmp(source.filename,char(options.KernelCache)), ...
            'arf_cbc:Prepared','KernelCache differs from the prepared source.');
    end
else
    root=fileparts(fileparts(mfilename('fullpath')));
    addpath(fullfile(root,'functions'));
    m=(n-1)/2;g=generatorp(n);perm=zeros(m,1);perm(1)=1;
    for j=2:m,perm(j)=mod(perm(j-1)*g,n);end
    perm=int64(min(perm,n-perm));
    if strlength(options.KernelCache)>0
        assert(~isempty(fieldnames(descriptor)),'arf_cbc:Cache','A cache requires a native Korobov descriptor.');
        folder=fileparts(options.KernelCache);metadata=jsondecode(fileread(fullfile(folder,'kernel.json')));
        assert(metadata.config.N==n && metadata.config.alpha==descriptor.alpha && metadata.bits>=options.Bits, ...
            'arf_cbc:Cache','Kernel cache metadata does not match.');
        fid=fopen(fullfile(folder,'perm.bin'),'r','ieee-le');assert(fid>=0);cache_perm=fread(fid,inf,'*int64');fclose(fid);
        assert(isequal(cache_perm,perm),'arf_cbc:Cache','Kernel cache permutation differs.');
        source=struct('kind','cache','filename',char(options.KernelCache),'precision','cached values rounded to working bits');
    elseif ~isempty(fieldnames(descriptor))
        source=struct('kind','native','alpha',descriptor.alpha,'precision','native guarded evaluation rounded to working bits');
    else
        values=omega([0;double(perm)/n]);
        text_values=iscellstr(values)||isstring(values);
        assert(numel(values)==m+1 && ((isa(values,'double') && isreal(values))||text_values), ...
            'arf_cbc:Kernel','omega must return real double values or decimal strings.');
        if isa(values,'double')
            warning('arf_cbc:DoubleKernel','omega returned double kernel values; use korobov_kernel_arf(alpha) for native high-precision kernel preparation.');
            precision='53-bit supplied kernel values';
        else
            precision='decimal strings supplied by callback';
            if isstring(values),values=cellstr(values);end
        end
        source=struct('kind','values','values',{values(:)},'precision',precision);
    end
    [id,preparation]=fastrank1pt_arf_mex('prepare',runtime.dll,n,s_max,options.Bits, ...
        options.Threads,perm,gamma,beta,source);
    preparation_seconds=toc(started);preparation.source=source;
    if isfield(preparation.source,'values'),preparation.source=rmfield(preparation.source,'values');end
    prepared=arf_cbc_plan(id,signature,preparation);
end
[z,e2,info]=fastrank1pt_arf_mex('run',prepared.Id,options.Audit);
info.call_seconds=toc(started);info.preparation_seconds=preparation_seconds;
info.prepared_reused=reused;info.kernel_source=source.kind;
info.kernel_method=prepared.Preparation.kernel_method;
if isfield(source,'precision'),info.kernel_precision=source.precision;end
info.audit=options.Audit;info.output_type='double after native high-precision evaluation';
end
function descriptor=kernel_descriptor(omega)
descriptor=struct();details=functions(omega);
if isfield(details,'workspace')
    for j=1:numel(details.workspace)
        workspace=details.workspace{j};
        if isfield(workspace,'arf_kernel_descriptor')
            candidate=workspace.arf_kernel_descriptor;
            if isstruct(candidate) && isfield(candidate,'tag') && strcmp(candidate.tag,'arf-korobov-v1')
                descriptor=candidate;return
            end
        end
    end
end
end
