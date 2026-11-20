function runtime=arf_cbc_runtime()
% Locate the standalone native bundle. No Python or Advanpix is consulted.
here=fileparts(mfilename('fullpath'));
assert(ispc && strcmp(mexext,'mexw64'),'arf_cbc:Platform', ...
    'The bundled CBC backend requires 64-bit Windows MATLAB.');
output=fullfile(here,'native','win64');
manifest=fullfile(output,'build.json');
assert(isfile(manifest),'arf_cbc:NotBuilt', ...
    'Copy the complete functions/cbc_precision/native/win64 bundle, including build.json, MEX and runtime DLLs.');
metadata=jsondecode(fileread(manifest));
assert(strcmp(metadata.flint_version,'3.3.1'),'arf_cbc:Version','This adapter requires FLINT 3.3.1.');
runtime.directory=fullfile(output,metadata.runtime_directory);
runtime.dll=fullfile(runtime.directory,metadata.flint_dll);
runtime.bundle=output;
runtime.manifest=manifest;
runtime.mex=fullfile(output,['fastrank1pt_arf_mex.' mexext]);
assert(isfile(runtime.dll),'arf_cbc:MissingRuntime','The native FLINT runtime is missing.');
assert(isfile(runtime.mex),'arf_cbc:NotBuilt','The bundled CBC MEX is missing. Copy the complete native/win64 folder.');
addpath(output,'-begin');
end
