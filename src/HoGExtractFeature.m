classdef HoGExtractFeature < ExtractFeature
    
    properties
    end
    
    methods
        function [Ad] = extract(A)
            Ad = extractHOGFeatures(A);
        end
    end
    
end
