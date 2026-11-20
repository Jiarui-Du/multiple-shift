function metadata=build_arf_cbc_mex(options)
% Build/package the native FLINT MEX; never starts Python or imports a pyd.
arguments
    options.CompilerRoot (1,1) string = ""
    options.FlintDirectory (1,1) string = ""
end
here=fileparts(mfilename('fullpath'));root=fileparts(fileparts(here));
output=fullfile(here,'native','win64');runtime=fullfile(output,'runtime');
if ~isfolder(runtime),mkdir(runtime);end
[~,loaded_mex]=inmem;
if any(contains(string(loaded_mex),'fastrank1pt_arf_mex'))
    status=fastrank1pt_arf_mex('status');assert(status.active_plans==0,'Release existing arf_cbc_plan objects before rebuilding.');
end
clear fastrank1pt_arf_mex
compiler_root=options.CompilerRoot;
if strlength(compiler_root)==0,compiler_root=fullfile(root,'build/cbc_precision/toolchain/mingw81');end
assert(isfile(fullfile(compiler_root,'bin/gcc.exe')),'A compatible MinGW C compiler is required.');
flint_dir=options.FlintDirectory;
if strlength(flint_dir)==0
    existing=dir(fullfile(runtime,'libflint*.dll'));
    if isscalar(existing)
        flint_dir=runtime;
    else
        flint_dir=fullfile(root,'build/cbc_precision/flint-packages/python_flint.libs');
    end
end
flint_files=dir(fullfile(flint_dir,'libflint*.dll'));assert(isscalar(flint_files),'Select the verified FLINT 3.3.1 directory.');
% These are independent native DLLs; the source directory name is not a runtime dependency.
native_files=dir(fullfile(flint_dir,'*.dll'));
for j=1:numel(native_files)
    source=fullfile(flint_dir,native_files(j).name);target=fullfile(runtime,native_files(j).name);
    if ~strcmpi(char(source),char(target)),copyfile(source,target);end
end
for name=["libgomp-1.dll","libwinpthread-1.dll","libgcc_s_seh-1.dll"]
    copyfile(fullfile(compiler_root,'bin',name),runtime);
end
old_root=getenv('MW_MINGW64_LOC');old_path=getenv('PATH');
restore=onCleanup(@()restore_environment(old_root,old_path));
setenv('PATH',[char(fullfile(compiler_root,'bin')) pathsep old_path]);
alias="";
if contains(compiler_root," ")
    for letter='Z':-1:'H'
        candidate=string(letter)+":";
        [status,~]=system(sprintf('subst %s "%s"',candidate,compiler_root));
        if status==0,alias=candidate;break;end
    end
    assert(strlength(alias)>0,'Could not create a temporary compiler drive alias.');
    alias_cleanup=onCleanup(@()system('subst '+alias+' /d'));
    setenv('MW_MINGW64_LOC',alias+'\');
else
    setenv('MW_MINGW64_LOC',compiler_root);
end
configs=mex.getCompilerConfigurations('C','Installed');
selected=find(contains(string({configs.Manufacturer}),"GNU"),1);
assert(~isempty(selected),'A compatible MinGW C MEX compiler was not detected.');
compiler=configs(selected);
flags={'CFLAGS=$CFLAGS -std=c11 -fopenmp -Wall -Wextra -Werror -fno-fast-math', ...
    'COPTIMFLAGS=-O3 -DNDEBUG -fno-fast-math','LDFLAGS=$LDFLAGS -fopenmp'};
mex('-R2018a','-v','-f',compiler.MexOpt,flags{:},'-outdir',output, ...
    '-output','fastrank1pt_arf_mex',fullfile(here,'fastrank1pt_arf_mex.c'));
metadata=struct('matlab',version,'compiler',compiler.Name,'compiler_version',compiler.Version, ...
    'flint_version','3.3.1','flint_dll',flint_files.name,'bits_default',113,'threads_default',8, ...
    'runtime_directory','runtime','python_required',false,'advanpix_required',false, ...
    'flint_source_directory',char(flint_dir),'compiler_root',char(compiler_root), ...
    'source','fastrank1pt_arf_mex.c + acf_point_native.c','flags',{flags});
fid=fopen(fullfile(output,'build.json'),'w');assert(fid>=0);
fprintf(fid,'%s\n',jsonencode(metadata,PrettyPrint=true));fclose(fid);
addpath(output);
fprintf('Built %s\n',fullfile(output,['fastrank1pt_arf_mex.' mexext]));
end
function restore_environment(compiler_root,path)
setenv('MW_MINGW64_LOC',compiler_root);setenv('PATH',path);
end
