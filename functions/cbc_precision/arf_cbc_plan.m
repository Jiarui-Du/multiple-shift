classdef arf_cbc_plan < handle
    % Native plan lease. Copies share ownership; delete releases the cache.
    properties (SetAccess=private)
        Id
        Signature
        Preparation
    end
    methods
        function obj=arf_cbc_plan(id,signature,preparation)
            obj.Id=id;obj.Signature=signature;obj.Preparation=preparation;
        end
        function delete(obj)
            for j=1:numel(obj)
                % isvalid is already false while MATLAB invokes delete.
                % Reading Id is permitted during deletion; a repeated delete
                % of an invalid handle may reject the property access.
                try
                    id=obj(j).Id;
                catch
                    continue
                end
                if ~isempty(id)
                    try
                        fastrank1pt_arf_mex('free',id);
                    catch
                        % MATLAB shutdown may have unloaded the gateway already.
                    end
                    obj(j).Id=[];
                end
            end
        end
    end
end
