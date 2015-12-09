classdef HoGExtractFeature < ExtractFeature
    
    properties
    end
    
    methods
        function [Ad] = extract(self, images)
            A = readImages(self, images);
            Ad = project(self, A);
        end
        
        function [Ad] = project(self, A)
            N = size(A,1);
            Ad = [];
            for i = 1 : N
                Ad(end+1,:) = extractHOGFeatures(reshape(A(i,:), ...
                    self.img_size));
            end
        end
    end
    
end
