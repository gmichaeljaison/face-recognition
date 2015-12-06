classdef HoGExtractFeature < ExtractFeature
    
    properties
    end
    
    methods
        function [Ad] = extract(self, images)
            A = readImages(self, images);
            Ad = extractHOGFeatures(A);
        end
        
        function [Ad] = project(self, A)
            Ad = extractHOGFeatures(A);
        end
    end
    
end
